#!/usr/bin/env python3

"""
KL-Divergence Method vs Ground Truth Experiment
Focus: How does KL-divergence optimization perform with different communication frequencies?
Key Question: What's the optimal merge interval for approaching ground truth?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
import seaborn as sns
import pandas as pd
from scipy.stats import entropy, mannwhitneyu
from scipy.optimize import minimize
import time
import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import sys
import socket
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def generate_consistent_seed(grid_size, n_agents, pattern, trial_id):
    """Generate consistent seed across sessions"""
    seed_string = f"{grid_size[0]}x{grid_size[1]}_{n_agents}agents_{pattern}_trial{trial_id}"
    hash_object = hashlib.sha256(seed_string.encode())
    hash_hex = hash_object.hexdigest()
    seed = int(hash_hex[:8], 16) % (2**31 - 1)
    return seed

class KLBeliefExperiment:
    """
    Focused experiment comparing KL-divergence method across merge intervals to ground truth
    """
    
    def __init__(self, grid_size=(20, 20), n_agents=4, alpha=0.1, beta=0.2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.alpha = alpha  # False positive rate
        self.beta = beta   # False negative rate
        self.n_states = grid_size[0] * grid_size[1]
        self.rows, self.cols = grid_size
        
    def compute_ground_truth_belief(self, all_observations: List[Dict], 
                                   initial_prior: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute ground truth belief using ALL observations from ALL agents
        This is what perfect information sharing would achieve
        """
        if initial_prior is None:
            belief = np.ones(self.n_states) / self.n_states
        else:
            belief = initial_prior.copy()
        
        # Process all observations in chronological order
        for obs_data in all_observations:
            pos = obs_data['position']
            obs = obs_data['observation']
            
            # Bayesian update with observation
            likelihood = np.ones(self.n_states)
            
            if obs == 1:  # Detection
                likelihood[pos] = 1 - self.beta
                likelihood[np.arange(self.n_states) != pos] = self.alpha
            else:  # No detection
                likelihood[pos] = self.beta
                likelihood[np.arange(self.n_states) != pos] = 1 - self.alpha
            
            belief = belief * likelihood
            belief = belief / (np.sum(belief) + 1e-10)
        
        return belief
    
    def merge_beliefs_kl(self, beliefs: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        KL-divergence minimization (your method)
        """
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        if weights is None:
            weights = np.ones(len(beliefs)) / len(beliefs)
        
        def objective(merged_flat):
            merged = np.clip(merged_flat, 1e-10, 1)
            merged = merged / np.sum(merged)
            
            total_div = 0
            for i, belief in enumerate(beliefs):
                belief_clip = np.clip(belief, 1e-10, 1)
                kl = np.sum(belief_clip * np.log(belief_clip / merged))
                total_div += weights[i] * kl
            
            return total_div
        
        # Initial guess: weighted average
        initial = np.average(beliefs, axis=0, weights=weights)
        initial = initial / np.sum(initial)
        
        # Optimization with your exact parameters
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(1e-10, 1) for _ in range(self.n_states)]
        
        result = minimize(
            objective,
            initial.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        merged = result.x
        return merged / np.sum(merged)
    
    def run_trial(self, merge_interval: int, target_trajectory: List[int], trial_seed: int) -> Dict:
        """Run a single trial with KL method at specified merge interval"""
        np.random.seed(trial_seed)
        
        # Initialize agent positions (spread across grid)
        positions = []
        for i in range(self.n_agents):
            region_r = (i // 2) * (self.rows // 2) + np.random.randint(self.rows // 2)
            region_c = (i % 2) * (self.cols // 2) + np.random.randint(self.cols // 2)
            positions.append(region_r * self.cols + region_c)
        
        # Initialize beliefs (independent for each agent)
        beliefs = [np.ones(self.n_states) / self.n_states for _ in range(self.n_agents)]
        
        # Storage for all observations (for ground truth)
        all_observations = []
        
        # Metrics tracking
        metrics = {
            'kl_divergence_to_truth': [],
            'js_divergence_to_truth': [],
            'entropy_merged': [],
            'entropy_truth': [],
            'target_prob_merged': [],
            'target_prob_truth': [],
            'belief_consensus': [],
            'merge_events': [],
            'communication_efficiency': []  # Track how much better we get with each merge
        }
        
        max_steps = min(len(target_trajectory), 300)
        
        for step in range(max_steps):
            target_pos = target_trajectory[step]
            
            # Simple agent movement (random walk)
            new_positions = []
            for i, pos in enumerate(positions):
                r, c = divmod(pos, self.cols)
                
                # Random movement
                moves = [(r, c)]  # Can stay
                if r > 0: moves.append((r-1, c))
                if r < self.rows-1: moves.append((r+1, c))
                if c > 0: moves.append((r, c-1))
                if c < self.cols-1: moves.append((r, c+1))
                
                new_r, new_c = moves[np.random.randint(len(moves))]
                new_pos = new_r * self.cols + new_c
                new_positions.append(new_pos)
            
            positions = new_positions
            
            # Make observations
            for i, pos in enumerate(positions):
                if pos == target_pos:
                    obs = 1 if np.random.random() > self.beta else 0
                else:
                    obs = 1 if np.random.random() < self.alpha else 0
                
                obs_data = {
                    'timestep': step,
                    'agent': i,
                    'position': pos,
                    'observation': obs
                }
                all_observations.append(obs_data)
                
                # Update individual belief
                likelihood = np.ones(self.n_states)
                if obs == 1:
                    likelihood[pos] = 1 - self.beta
                    likelihood[np.arange(self.n_states) != pos] = self.alpha
                else:
                    likelihood[pos] = self.beta
                    likelihood[np.arange(self.n_states) != pos] = 1 - self.alpha
                
                beliefs[i] = beliefs[i] * likelihood
                beliefs[i] = beliefs[i] / (np.sum(beliefs[i]) + 1e-10)
            
            # Track KL divergence before potential merge
            current_merged = self.merge_beliefs_kl(beliefs)
            kl_before_merge = None
            
            # Merge if scheduled
            if merge_interval > 0 and step > 0 and step % merge_interval == 0:
                kl_before_merge = self._compute_kl_to_truth(current_merged, all_observations)
                
                merged = self.merge_beliefs_kl(beliefs)
                beliefs = [merged.copy() for _ in range(self.n_agents)]
                
                kl_after_merge = self._compute_kl_to_truth(merged, all_observations)
                
                metrics['merge_events'].append({
                    'step': step,
                    'kl_before': kl_before_merge,
                    'kl_after': kl_after_merge,
                    'improvement': kl_before_merge - kl_after_merge,
                    'entropy_before': entropy(current_merged),
                    'entropy_after': entropy(merged)
                })
                
                current_merged = merged
            
            # Compute ground truth
            ground_truth = self.compute_ground_truth_belief(all_observations)
            
            # Calculate metrics
            gt_clipped = np.clip(ground_truth, 1e-10, 1)
            merged_clipped = np.clip(current_merged, 1e-10, 1)
            kl_div = np.sum(gt_clipped * np.log(gt_clipped / merged_clipped))
            metrics['kl_divergence_to_truth'].append(kl_div)
            
            # JS divergence (symmetric)
            m = 0.5 * (ground_truth + current_merged)
            js_div = 0.5 * np.sum(ground_truth * np.log(np.clip(ground_truth / m, 1e-10, 1e10))) + \
                     0.5 * np.sum(current_merged * np.log(np.clip(current_merged / m, 1e-10, 1e10)))
            metrics['js_divergence_to_truth'].append(js_div)
            
            # Entropies
            metrics['entropy_merged'].append(entropy(current_merged))
            metrics['entropy_truth'].append(entropy(ground_truth))
            
            # Target probabilities
            metrics['target_prob_merged'].append(current_merged[target_pos])
            metrics['target_prob_truth'].append(ground_truth[target_pos])
            
            # Consensus (correlation between agent beliefs)
            if len(beliefs) > 1:
                correlations = []
                for i in range(len(beliefs)):
                    for j in range(i+1, len(beliefs)):
                        corr = np.corrcoef(beliefs[i], beliefs[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                metrics['belief_consensus'].append(np.mean(correlations) if correlations else 0)
            else:
                metrics['belief_consensus'].append(1.0)
        
        # Final calculations
        predicted_target = np.argmax(current_merged)
        actual_target = target_trajectory[-1]
        
        # Prediction error (distance)
        pred_r, pred_c = divmod(predicted_target, self.cols)
        actual_r, actual_c = divmod(actual_target, self.cols)
        prediction_error = np.sqrt((pred_r - actual_r)**2 + (pred_c - actual_c)**2)
        
        return {
            'merge_interval': merge_interval,
            'trial_seed': trial_seed,
            
            # Performance metrics
            'avg_kl_to_truth': np.mean(metrics['kl_divergence_to_truth']),
            'final_kl_to_truth': metrics['kl_divergence_to_truth'][-1],
            'avg_js_to_truth': np.mean(metrics['js_divergence_to_truth']),
            'final_js_to_truth': metrics['js_divergence_to_truth'][-1],
            'avg_target_prob_merged': np.mean(metrics['target_prob_merged']),
            'avg_target_prob_truth': np.mean(metrics['target_prob_truth']),
            'final_target_prob_merged': metrics['target_prob_merged'][-1],
            'final_target_prob_truth': metrics['target_prob_truth'][-1],
            'avg_consensus': np.mean(metrics['belief_consensus']),
            'prediction_error': prediction_error,
            'n_merges': len(metrics['merge_events']),
            'communication_efficiency': np.mean([e['improvement'] for e in metrics['merge_events']]) if metrics['merge_events'] else 0,
            
            # Full time series for detailed analysis
            'time_series': metrics
        }
    
    def _compute_kl_to_truth(self, belief: np.ndarray, all_observations: List[Dict]) -> float:
        """Helper to compute KL divergence to ground truth"""
        ground_truth = self.compute_ground_truth_belief(all_observations)
        gt_clipped = np.clip(ground_truth, 1e-10, 1)
        belief_clipped = np.clip(belief, 1e-10, 1)
        return np.sum(gt_clipped * np.log(gt_clipped / belief_clipped))


def create_kl_focused_visualizations(results: Dict, output_dir: Path):
    """
    Create focused visualizations for KL method across merge intervals
    """
    print(f"Creating KL-focused visualizations in {output_dir}...")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to DataFrame - focus only on KL method
    df = flatten_kl_results_to_dataframe(results)
    if df.empty:
        print("No KL data available for visualization")
        return
    
    # Create the main comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    merge_intervals = sorted(df['merge_interval'].unique())
    interval_colors = plt.cm.viridis(np.linspace(0, 1, len(merge_intervals)))
    
    # 1. Main Performance Comparison Across Intervals
    ax = fig.add_subplot(gs[0, :2])
    
    # Box plot showing performance distribution for each interval
    box_data = [df[df['merge_interval'] == interval]['avg_kl_to_truth'].values 
                for interval in merge_intervals]
    
    bp = ax.boxplot(box_data, labels=[f"Interval {int(i) if i != float('inf') else '∞'}" for i in merge_intervals],
                    patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], interval_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('KL Divergence to Ground Truth')
    ax.set_title('KL Method Performance vs Communication Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, interval in enumerate(merge_intervals):
        mean_val = df[df['merge_interval'] == interval]['avg_kl_to_truth'].mean()
        ax.text(i+1, mean_val, f'{mean_val:.4f}', ha='center', va='bottom', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 2. Convergence Rate Analysis
    ax = fig.add_subplot(gs[0, 2:])
    
    for i, interval in enumerate(merge_intervals):
        interval_data = df[df['merge_interval'] == interval]
        
        # Get time series data from first available trial
        sample_trial = None
        for grid_results in results.values():
            for agent_results in grid_results.values():
                for pattern_results in agent_results.values():
                    for interval_key, interval_trials in pattern_results.items():
                        if 'kl_divergence' in interval_trials:
                            for trial in interval_trials['kl_divergence']:
                                if (trial['merge_interval'] == interval and 
                                    'time_series' in trial and 
                                    trial['time_series']['kl_divergence_to_truth']):
                                    sample_trial = trial
                                    break
                            if sample_trial: break
                        if sample_trial: break
                    if sample_trial: break
                if sample_trial: break
            if sample_trial: break
        
        if sample_trial:
            time_series = sample_trial['time_series']['kl_divergence_to_truth']
            steps = range(len(time_series))
            
            label = f"Interval {int(interval)}" if interval != float('inf') else "No Merge (∞)"
            ax.plot(steps, time_series, color=interval_colors[i], linewidth=2, 
                   label=label, alpha=0.8)
            
            # Mark merge events
            if 'merge_events' in sample_trial['time_series']:
                for event in sample_trial['time_series']['merge_events']:
                    ax.axvline(event['step'], color=interval_colors[i], alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('KL Divergence to Ground Truth')
    ax.set_title('Convergence to Ground Truth Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Communication Efficiency Analysis
    ax = fig.add_subplot(gs[1, 0])
    
    # Calculate improvement per merge event
    efficiency_data = []
    for interval in merge_intervals:
        if interval != float('inf'):
            interval_efficiency = df[df['merge_interval'] == interval]['communication_efficiency'].mean()
            efficiency_data.append(interval_efficiency)
        else:
            efficiency_data.append(0)  # No merges = no improvement
    
    bars = ax.bar(range(len(merge_intervals)), efficiency_data, 
                  color=interval_colors, alpha=0.7)
    ax.set_xticks(range(len(merge_intervals)))
    ax.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
    ax.set_xlabel('Merge Interval')
    ax.set_ylabel('Avg KL Improvement per Merge')
    ax.set_title('Communication Efficiency')
    
    # Add value labels
    for bar, value in zip(bars, efficiency_data):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Optimal Frequency Analysis
    ax = fig.add_subplot(gs[1, 1])
    
    # Plot mean performance vs frequency (1/interval)
    frequencies = [1/interval if interval != float('inf') else 0 for interval in merge_intervals]
    mean_performance = [df[df['merge_interval'] == interval]['avg_kl_to_truth'].mean() 
                       for interval in merge_intervals]
    
    ax.scatter(frequencies, mean_performance, c=interval_colors, s=100, alpha=0.8)
    
    # Fit a curve to show trend
    if len(frequencies) > 2:
        # Exclude the no-merge case for curve fitting
        freq_fit = [f for f in frequencies if f > 0]
        perf_fit = [p for f, p in zip(frequencies, mean_performance) if f > 0]
        
        if len(freq_fit) > 1:
            z = np.polyfit(freq_fit, perf_fit, 2)
            p = np.poly1d(z)
            freq_smooth = np.linspace(min(freq_fit), max(freq_fit), 100)
            ax.plot(freq_smooth, p(freq_smooth), 'r--', alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Communication Frequency (1/interval)')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Performance vs Communication Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for each point
    for freq, perf, interval in zip(frequencies, mean_performance, merge_intervals):
        label = f"{int(interval)}" if interval != float('inf') else "∞"
        ax.annotate(label, (freq, perf), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 5. Target Probability Tracking
    ax = fig.add_subplot(gs[1, 2:])
    
    for i, interval in enumerate(merge_intervals):
        # Get sample time series for target probability
        sample_trial = None
        for grid_results in results.values():
            for agent_results in grid_results.values():
                for pattern_results in agent_results.values():
                    for interval_key, interval_trials in pattern_results.items():
                        if 'kl_divergence' in interval_trials:
                            for trial in interval_trials['kl_divergence']:
                                if (trial['merge_interval'] == interval and 
                                    'time_series' in trial and 
                                    trial['time_series']['target_prob_merged']):
                                    sample_trial = trial
                                    break
                            if sample_trial: break
                        if sample_trial: break
                    if sample_trial: break
                if sample_trial: break
            if sample_trial: break
        
        if sample_trial:
            ts = sample_trial['time_series']
            steps = range(len(ts['target_prob_merged']))
            
            label = f"KL-{int(interval)}" if interval != float('inf') else "KL-∞"
            ax.plot(steps, ts['target_prob_merged'], color=interval_colors[i], 
                   linewidth=2, label=label, alpha=0.8)
            
            # Show ground truth as reference (only once)
            if i == 0:
                ax.plot(steps, ts['target_prob_truth'], 'k--', linewidth=2, 
                       label='Ground Truth', alpha=0.9)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Probability at True Target')
    ax.set_title('Target Probability Evolution by Merge Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    for interval in merge_intervals:
        interval_data = df[df['merge_interval'] == interval]
        
        summary_data.append([
            f"{int(interval)}" if interval != float('inf') else "∞",
            f"{interval_data['avg_kl_to_truth'].mean():.5f}",
            f"{interval_data['avg_kl_to_truth'].std():.5f}",
            f"{interval_data['final_target_prob_merged'].mean():.3f}",
            f"{interval_data['prediction_error'].mean():.2f}",
            f"{interval_data['n_merges'].mean():.1f}"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Interval', 'Mean KL', 'Std KL', 'Target Prob', 'Pred Error', 'N Merges'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0.3, 1, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            table[(i+1, j)].set_facecolor(interval_colors[i])
            table[(i+1, j)].set_alpha(0.3)
    
    ax.set_title('Performance Summary by Merge Interval', fontsize=12, fontweight='bold', y=0.9)
    
    # 7. Key Findings Text
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    
    # Analyze results for key findings
    best_interval = df.groupby('merge_interval')['avg_kl_to_truth'].mean().idxmin()
    best_performance = df.groupby('merge_interval')['avg_kl_to_truth'].mean().min()
    worst_interval = df.groupby('merge_interval')['avg_kl_to_truth'].mean().idxmax()
    worst_performance = df.groupby('merge_interval')['avg_kl_to_truth'].mean().max()
    
    improvement_vs_no_merge = ((df[df['merge_interval'] == float('inf')]['avg_kl_to_truth'].mean() - 
                               best_performance) / 
                              df[df['merge_interval'] == float('inf')]['avg_kl_to_truth'].mean() * 100)
    
    findings_text = f"""KL-DIVERGENCE METHOD ANALYSIS
{'='*50}

Key Findings:
• Best merge interval: {int(best_interval) if best_interval != float('inf') else '∞'}
• Best performance: {best_performance:.5f} KL divergence
• Worst performance: {worst_performance:.5f} KL divergence  
• Improvement over no merging: {improvement_vs_no_merge:.1f}%

Communication Efficiency:
• Total trials analyzed: {len(df):,}
• Merge intervals tested: {', '.join([str(int(i)) if i != float('inf') else '∞' for i in sorted(merge_intervals)])}

Conclusion:
The optimal communication frequency for the KL-divergence
method appears to be interval {int(best_interval) if best_interval != float('inf') else '∞'}, achieving
{best_performance:.5f} KL divergence to ground truth.

{'More frequent' if best_interval < 10 else 'Less frequent'} communication 
{'improves' if improvement_vs_no_merge > 0 else 'does not improve'} performance significantly.
"""
    
    ax.text(0.05, 0.95, findings_text, transform=ax.transAxes, 
           fontfamily='monospace', fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # Main title
    plt.suptitle('KL-Divergence Method: Optimal Communication Frequency Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    main_plot_path = output_dir / 'kl_method_communication_analysis.png'
    plt.tight_layout()
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"KL-focused visualization saved: {main_plot_path}")
    
    # Create additional detailed time series plot
    create_detailed_time_series_plot(results, output_dir)
    
    return main_plot_path


def flatten_kl_results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert nested results to flat DataFrame focusing on KL method only"""
    data_rows = []
    
    for grid_key, grid_results in results.items():
        for agent_key, agent_results in grid_results.items():
            try:
                n_agents = int(agent_key.split('_')[0])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse agent number from {agent_key}")
                continue
                
            for pattern, pattern_results in agent_results.items():
                for interval_key, interval_data in pattern_results.items():
                    
                    # Focus only on KL divergence method
                    if isinstance(interval_data, dict) and 'kl_divergence' in interval_data:
                        trials = interval_data['kl_divergence']
                        
                        for trial in trials:
                            if isinstance(trial, dict) and 'avg_kl_to_truth' in trial:
                                data_rows.append({
                                    'grid_size': grid_key,
                                    'n_agents': n_agents,
                                    'pattern': pattern,
                                    'interval_strategy': interval_key,
                                    'merge_interval': trial.get('merge_interval', float('inf')),
                                    'avg_kl_to_truth': trial.get('avg_kl_to_truth', np.nan),
                                    'final_kl_to_truth': trial.get('final_kl_to_truth', np.nan),
                                    'avg_target_prob_merged': trial.get('avg_target_prob_merged', np.nan),
                                    'avg_target_prob_truth': trial.get('avg_target_prob_truth', np.nan),
                                    'final_target_prob_merged': trial.get('final_target_prob_merged', np.nan),
                                    'final_target_prob_truth': trial.get('final_target_prob_truth', np.nan),
                                    'prediction_error': trial.get('prediction_error', np.nan),
                                    'avg_consensus': trial.get('avg_consensus', np.nan),
                                    'n_merges': trial.get('n_merges', 0),
                                    'communication_efficiency': trial.get('communication_efficiency', 0)
                                })
    
    if not data_rows:
        print("ERROR: No valid KL method data found in results!")
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows)
    df = df.dropna(subset=['avg_kl_to_truth'])
    
    print(f"Flattened {len(df)} KL method data points")
    print(f"Merge intervals: {sorted(df['merge_interval'].unique())}")
    print(f"Grid sizes: {list(df['grid_size'].unique())}")
    print(f"Agent numbers: {sorted(df['n_agents'].unique())}")
    
    return df


def create_detailed_time_series_plot(results: Dict, output_dir: Path):
    """Create detailed time series plots for each merge interval"""
    
    # Collect time series data for each interval
    interval_data = {}
    
    for grid_results in results.values():
        for agent_results in grid_results.values():
            for pattern_results in agent_results.values():
                for interval_key, interval_trials in pattern_results.items():
                    if 'kl_divergence' in interval_trials:
                        for trial in interval_trials['kl_divergence']:
                            if 'time_series' in trial and trial['time_series']['kl_divergence_to_truth']:
                                interval = trial['merge_interval']
                                if interval not in interval_data:
                                    interval_data[interval] = []
                                interval_data[interval].append(trial['time_series'])
    
    if not interval_data:
        print("No time series data found for detailed plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    intervals = sorted(interval_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))
    
    # Plot 1: KL Divergence Evolution
    ax = axes[0]
    for i, interval in enumerate(intervals):
        # Average across trials for this interval
        all_series = interval_data[interval]
        max_len = max(len(series['kl_divergence_to_truth']) for series in all_series)
        
        # Pad shorter series with their last value
        padded_series = []
        for series in all_series:
            kl_series = series['kl_divergence_to_truth']
            if len(kl_series) < max_len:
                kl_series = kl_series + [kl_series[-1]] * (max_len - len(kl_series))
            padded_series.append(kl_series)
        
        mean_series = np.mean(padded_series, axis=0)
        std_series = np.std(padded_series, axis=0)
        steps = range(len(mean_series))
        
        label = f"Interval {int(interval)}" if interval != float('inf') else "No Merge (∞)"
        ax.plot(steps, mean_series, color=colors[i], linewidth=2, label=label)
        ax.fill_between(steps, mean_series - std_series, mean_series + std_series, 
                       color=colors[i], alpha=0.2)
        
        # Mark typical merge points
        if interval != float('inf') and interval > 0:
            merge_points = range(int(interval), len(steps), int(interval))
            for mp in merge_points[:5]:  # Show first 5 merge points
                ax.axvline(mp, color=colors[i], alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('KL Divergence to Ground Truth')
    ax.set_title('KL Divergence Evolution by Merge Interval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Target Probability Evolution
    ax = axes[1]
    for i, interval in enumerate(intervals):
        all_series = interval_data[interval]
        max_len = max(len(series['target_prob_merged']) for series in all_series)
        
        padded_series = []
        for series in all_series:
            prob_series = series['target_prob_merged']
            if len(prob_series) < max_len:
                prob_series = prob_series + [prob_series[-1]] * (max_len - len(prob_series))
            padded_series.append(prob_series)
        
        mean_series = np.mean(padded_series, axis=0)
        std_series = np.std(padded_series, axis=0)
        steps = range(len(mean_series))
        
        label = f"KL-{int(interval)}" if interval != float('inf') else "KL-∞"
        ax.plot(steps, mean_series, color=colors[i], linewidth=2, label=label)
        ax.fill_between(steps, mean_series - std_series, mean_series + std_series, 
                       color=colors[i], alpha=0.2)
    
    # Add ground truth reference
    if interval_data:
        sample_truth = list(interval_data.values())[0][0]['target_prob_truth']
        ax.plot(range(len(sample_truth)), sample_truth, 'k--', linewidth=2, 
               label='Ground Truth', alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Probability at True Target')
    ax.set_title('Target Probability Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Consensus Evolution
    ax = axes[2]
    for i, interval in enumerate(intervals):
        all_series = interval_data[interval]
        max_len = max(len(series['belief_consensus']) for series in all_series)
        
        padded_series = []
        for series in all_series:
            consensus_series = series['belief_consensus']
            if len(consensus_series) < max_len:
                consensus_series = consensus_series + [consensus_series[-1]] * (max_len - len(consensus_series))
            padded_series.append(consensus_series)
        
        mean_series = np.mean(padded_series, axis=0)
        steps = range(len(mean_series))
        
        label = f"Interval {int(interval)}" if interval != float('inf') else "No Merge (∞)"
        ax.plot(steps, mean_series, color=colors[i], linewidth=2, label=label)
        
        # Mark merge events with consensus jumps
        if interval != float('inf') and interval > 0:
            merge_points = range(int(interval), len(steps), int(interval))
            for mp in merge_points[:5]:
                ax.axvline(mp, color=colors[i], alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Belief Consensus')
    ax.set_title('Agent Consensus Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Communication Impact Analysis
    ax = axes[3]
    
    # Show improvement gained at each merge event
    merge_improvements = {interval: [] for interval in intervals if interval != float('inf')}
    
    for interval in intervals:
        if interval == float('inf'):
            continue
            
        for series in interval_data[interval]:
            if 'merge_events' in series:
                improvements = [event['improvement'] for event in series['merge_events'] 
                              if event['improvement'] > 0]
                merge_improvements[interval].extend(improvements)
    
    # Box plot of improvements
    improvement_data = []
    improvement_labels = []
    
    for interval in sorted(merge_improvements.keys()):
        if merge_improvements[interval]:
            improvement_data.append(merge_improvements[interval])
            improvement_labels.append(f"Interval {int(interval)}")
    
    if improvement_data:
        bp = ax.boxplot(improvement_data, labels=improvement_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(improvement_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('KL Improvement per Merge')
    ax.set_title('Communication Impact Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Time Series Analysis: KL Method Communication Patterns', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    detailed_plot_path = output_dir / 'kl_detailed_time_series.png'
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed time series plot saved: {detailed_plot_path}")


class DistributedExperimentManager:
    """Manages distributed execution focusing on KL method only"""
    
    def __init__(self, config: Dict, checkpoint_dir: str = "checkpoints", 
                 results_dir: str = "results", max_workers: int = None):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 64)
        self.max_workers = max_workers
        
        print(f"Experiment manager initialized with {self.max_workers} workers")
    
    def generate_all_tasks(self) -> List[Dict]:
        """Generate all trial tasks - focusing only on KL method"""
        tasks = []
        
        for grid_size in self.config['grid_sizes']:
            for n_agents in self.config['n_agents_list']:
                for pattern in self.config['target_patterns']:
                    for trial_id in range(self.config['n_trials']):
                        trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                        
                        for merge_interval in self.config['merge_intervals']:
                            task = {
                                'grid_size': grid_size,
                                'n_agents': n_agents,
                                'pattern': pattern,
                                'trial_id': trial_id,
                                'merge_interval': merge_interval,
                                'trial_seed': trial_seed,
                                'config': self.config
                            }
                            tasks.append(task)
        
        print(f"Generated {len(tasks)} total tasks (KL method only)")
        return tasks
    
    def get_checkpoint_path(self, task: Dict) -> str:
        """Get checkpoint file path for task"""
        grid_str = f"{task['grid_size'][0]}x{task['grid_size'][1]}"
        interval_str = 'inf' if task['merge_interval'] == float('inf') else str(task['merge_interval'])
        filename = (f"grid{grid_str}_agents{task['n_agents']}_{task['pattern']}_"
                   f"trial{task['trial_id']}_interval{interval_str}_"
                   f"kl_method_seed{task['trial_seed']}.pkl")
        return str(self.checkpoint_dir / filename)
    
    def filter_incomplete_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Filter out completed tasks"""
        incomplete_tasks = []
        completed_count = 0
        
        for task in tasks:
            checkpoint_path = self.get_checkpoint_path(task)
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'rb') as f:
                        result = pickle.load(f)
                    # Validate result has required fields
                    required_fields = ['avg_kl_to_truth', 'final_kl_to_truth', 'merge_interval']
                    if all(field in result for field in required_fields):
                        completed_count += 1
                        continue
                except:
                    pass
            
            incomplete_tasks.append(task)
        
        print(f"Found {completed_count} completed tasks")
        print(f"Remaining tasks: {len(incomplete_tasks)}")
        return incomplete_tasks
    
    def run_distributed_experiment(self):
        """Run the complete experiment"""
        print("Starting KL-focused belief merging experiment")
        start_time = time.time()
        
        # Generate and filter tasks
        all_tasks = self.generate_all_tasks()
        remaining_tasks = self.filter_incomplete_tasks(all_tasks)
        
        if not remaining_tasks:
            print("All tasks already completed!")
            return self.collect_results()
        
        # Run tasks in parallel
        print(f"Running {len(remaining_tasks)} tasks with {self.max_workers} workers")
        
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_kl_task, task): task 
                for task in remaining_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        completed += 1
                        if completed % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
                            print(f"Progress: {completed}/{len(remaining_tasks)} "
                                  f"({100*completed/len(remaining_tasks):.1f}%) "
                                  f"Rate: {rate:.1f} tasks/sec, ETA: {eta/60:.1f} min")
                    else:
                        failed += 1
                
                except Exception as e:
                    failed += 1
                    print(f"Task failed: {e}")
        
        total_time = time.time() - start_time
        print(f"Experiment completed in {total_time/60:.1f} minutes")
        print(f"Completed: {completed}, Failed: {failed}")
        
        return self.collect_results()
    
    def collect_results(self):
        """Collect all results and trigger visualization"""
        print("Collecting results...")
        
        all_results = {}
        
        for grid_size in self.config['grid_sizes']:
            grid_key = f"{grid_size[0]}x{grid_size[1]}"
            all_results[grid_key] = {}
            
            for n_agents in self.config['n_agents_list']:
                agent_key = f"{n_agents}_agents"
                all_results[grid_key][agent_key] = {}
                
                for pattern in self.config['target_patterns']:
                    pattern_results = {}
                    
                    for merge_interval in self.config['merge_intervals']:
                        interval_results = {'kl_divergence': []}
                        
                        for trial_id in range(self.config['n_trials']):
                            trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                            
                            task = {
                                'grid_size': grid_size,
                                'n_agents': n_agents,
                                'pattern': pattern,
                                'trial_id': trial_id,
                                'merge_interval': merge_interval,
                                'trial_seed': trial_seed
                            }
                            
                            checkpoint_path = self.get_checkpoint_path(task)
                            
                            if os.path.exists(checkpoint_path):
                                try:
                                    with open(checkpoint_path, 'rb') as f:
                                        result = pickle.load(f)
                                    interval_results['kl_divergence'].append(result)
                                except Exception as e:
                                    print(f"Failed to load {checkpoint_path}: {e}")
                        
                        # Store results with proper key naming
                        if merge_interval == 1:
                            key = 'immediate_merge'
                        elif merge_interval == float('inf'):
                            key = 'no_merge'
                        else:
                            key = f'interval_{merge_interval}'
                        
                        pattern_results[key] = interval_results
                    
                    all_results[grid_key][agent_key][pattern] = pattern_results
        
        # Save consolidated results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        consolidated_path = self.results_dir / f"kl_focused_results_{timestamp}.pkl"
        with open(consolidated_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"Results saved to {consolidated_path}")
        
        # Check if we have enough data for visualization
        total_trials = self._count_total_trials(all_results)
        
        if total_trials >= 10:
            print(f"Generating visualizations with {total_trials} data points...")
            try:
                output_dir = self.results_dir / "kl_analysis"
                create_kl_focused_visualizations(all_results, output_dir)
                print("Visualizations generated successfully!")
            except Exception as e:
                print(f"Visualization failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Not enough data for visualization (found {total_trials} trials, need at least 10)")
        
        return all_results
    
    def _count_total_trials(self, results: Dict) -> int:
        """Count total number of trials in results"""
        total = 0
        for grid_results in results.values():
            for agent_results in grid_results.values():
                for pattern_results in agent_results.values():
                    for interval_results in pattern_results.values():
                        if 'kl_divergence' in interval_results:
                            total += len(interval_results['kl_divergence'])
        return total


def run_single_kl_task(task: Dict) -> Optional[Dict]:
    """Run a single KL trial task"""
    checkpoint_path = Path(task['config']['checkpoint_dir']) / get_kl_checkpoint_filename(task)
    
    # Check if already completed
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except:
            checkpoint_path.unlink()  # Remove corrupted file
    
    try:
        # Create experiment
        experiment = KLBeliefExperiment(
            grid_size=task['grid_size'],
            n_agents=task['n_agents'],
            alpha=task['config']['alpha'],
            beta=task['config']['beta']
        )
        
        # Generate target trajectory
        np.random.seed(task['trial_seed'])
        
        # Simple random walk trajectory
        target_pos = np.random.randint(experiment.n_states)
        trajectory = [target_pos]
        
        for _ in range(task['config']['max_steps']):
            r, c = divmod(target_pos, experiment.cols)
            moves = [target_pos]  # Can stay
            if r > 0: moves.append(target_pos - experiment.cols)
            if r < experiment.rows-1: moves.append(target_pos + experiment.cols)
            if c > 0: moves.append(target_pos - 1)
            if c < experiment.cols-1: moves.append(target_pos + 1)
            target_pos = np.random.choice(moves)
            trajectory.append(target_pos)
        
        # Run the trial
        result = experiment.run_trial(
            task['merge_interval'], trajectory, task['trial_seed']
        )
        
        # Add metadata
        result['metadata'] = {
            'grid_size': task['grid_size'],
            'n_agents': task['n_agents'],
            'pattern': task['pattern'],
            'trial_id': task['trial_id'],
            'merge_interval': task['merge_interval'],
            'completion_time': datetime.now().isoformat()
        }
        
        # Save result atomically
        temp_path = str(checkpoint_path) + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(result, f)
        os.rename(temp_path, checkpoint_path)
        
        return result
        
    except Exception as e:
        print(f"Task failed: {e}")
        return None


def get_kl_checkpoint_filename(task: Dict) -> str:
    """Generate checkpoint filename for KL task"""
    grid_str = f"{task['grid_size'][0]}x{task['grid_size'][1]}"
    interval_str = str(task['merge_interval']) if task['merge_interval'] != float('inf') else 'inf'
    return (f"grid{grid_str}_agents{task['n_agents']}_{task['pattern']}_"
            f"trial{task['trial_id']}_interval{interval_str}_"
            f"kl_method_seed{task['trial_seed']}.pkl")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='KL-Focused Belief Merging vs Ground Truth Experiment')
    parser.add_argument('--config-file', type=str, help='JSON config file path')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Default configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'grid_sizes': [[10, 10], [20, 20]],
            'n_agents_list': [3, 4],
            'alpha': 0.1,
            'beta': 0.2,
            'n_trials': 50,
            'max_steps': 200,
            'merge_intervals': [1, 10, 25, float('inf')],
            'target_patterns': ['random'],
            'checkpoint_dir': args.checkpoint_dir
        }
    
    print("=" * 80)
    print("KL-DIVERGENCE METHOD vs GROUND TRUTH EXPERIMENT")
    print("=" * 80)
    print(f"Focus: Testing optimal communication frequency for KL method")
    print(f"Grid sizes: {config['grid_sizes']}")
    print(f"Agent numbers: {config['n_agents_list']}")
    print(f"Merge intervals: {config['merge_intervals']}")
    print(f"Trials per configuration: {config['n_trials']}")
    
    # Create and run experiment
    manager = DistributedExperimentManager(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        max_workers=args.max_workers
    )
    
    # Run experiment
    results = manager.run_distributed_experiment()
    
    if results:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved in {args.results_dir}")
        print("Generated files:")
        print("  - kl_method_communication_analysis.png: Main analysis plots")
        print("  - kl_detailed_time_series.png: Detailed time evolution")
        print("  - kl_focused_results_*.pkl: Raw data for further analysis")
    else:
        print("Experiment failed or was interrupted")


if __name__ == "__main__":
    main()