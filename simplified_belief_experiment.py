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
        # Add visit counting
        self.visit_counts = np.zeros((n_agents, self.n_states))
        
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
        #print("we're in standard_KL")
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
                # kl = np.sum(belief_clip * np.log(belief_clip / merged))
                kl = np.sum(merged * np.log(merged / belief_clip))
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
    
    def merge_beliefs_visit_weighted(self, beliefs: List[np.ndarray], visit_counts: np.ndarray) -> np.ndarray:
        """Visit-weighted KL-divergence minimization"""
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        n_agents = len(beliefs)
        
        # CORRECTED: State-specific weights
        weights_matrix = np.zeros((n_agents, self.n_states))
        
        for s in range(self.n_states):
            # For each state, compute how much each agent visited it
            visits_at_s = visit_counts[:, s] + 1  # Add 1 to avoid zeros
            total_visits_at_s = np.sum(visits_at_s)
            
            # Normalize: weights sum to 1 across agents for this state
            weights_matrix[:, s] = visits_at_s / total_visits_at_s
        
        # Verify weights sum to 1 for each state
        assert np.allclose(np.sum(weights_matrix, axis=0), 1.0), \
            "Weights should sum to 1 for each state"
        
        def objective(merged_flat):
            merged = np.clip(merged_flat, 1e-10, 1)
            merged = merged / np.sum(merged)
            
            total_div = 0
            for i, belief in enumerate(beliefs):
                belief_clip = np.clip(belief, 1e-10, 1)
                # State-specific weighted KL
                # For each state s: w_i(s) * belief_i(s) * log(belief_i(s) / merged(s))
                kl_per_state = belief_clip * np.log(belief_clip / merged)
                weighted_kl = np.sum(weights_matrix[i] * kl_per_state)
                total_div += weighted_kl
            
            return total_div
        
        # Initial guess: weighted average by total visits per agent
        agent_weights = np.sum(visit_counts + 1, axis=1)
        agent_weights = agent_weights / np.sum(agent_weights)
        initial = np.average(beliefs, axis=0, weights=agent_weights)
        initial = initial / np.sum(initial)
        
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
    def merge_beliefs_observation_weighted(self, beliefs: List[np.ndarray], 
                                    observation_counts: np.ndarray) -> np.ndarray:
        """Weight by actual observations (detections) at each position"""
        #print("we're in observation weighted KL")
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        # Calculate weights based on positive observations
        total_obs = np.sum(observation_counts) + len(beliefs)  # Add small constant
        weights = (np.sum(observation_counts, axis=1) + 1) / total_obs
        
        def objective(merged_flat):
            merged = np.clip(merged_flat, 1e-10, 1)
            merged = merged / np.sum(merged)
            
            total_div = 0
            for i, belief in enumerate(beliefs):
                belief_clip = np.clip(belief, 1e-10, 1)
                # kl = np.sum(belief_clip * np.log(belief_clip / merged))
                kl = np.sum(merged * np.log(merged / belief_clip))
                total_div += weights[i] * kl
            
            return total_div
        
        initial = np.average(beliefs, axis=0, weights=weights)
        initial = initial / np.sum(initial)
        
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

    def merge_beliefs_entropy_weighted(self, beliefs: List[np.ndarray]) -> np.ndarray:
        """Weight by inverse entropy (more certain beliefs get higher weight)"""
        #print("we're in entropy KL")
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        # Calculate entropy-based weights
        entropies = np.array([entropy(belief) for belief in beliefs])
        # Inverse entropy weighting (lower entropy = higher weight)
        weights = 1.0 / (1.0 + entropies)
        weights = weights / np.sum(weights)
        
        def objective(merged_flat):
            merged = np.clip(merged_flat, 1e-10, 1)
            merged = merged / np.sum(merged)
            
            total_div = 0
            for i, belief in enumerate(beliefs):
                belief_clip = np.clip(belief, 1e-10, 1)
                # kl = np.sum(belief_clip * np.log(belief_clip / merged))
                kl = np.sum(merged * np.log(merged / belief_clip))
                total_div += weights[i] * kl
            
            return total_div
        
        initial = np.average(beliefs, axis=0, weights=weights)
        initial = initial / np.sum(initial)
        
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

    def merge_beliefs_confidence_weighted(self, beliefs: List[np.ndarray], 
                                     lambda_penalty: float = 0.1,
                                     eta_threshold: float = 0.3) -> np.ndarray:
        """Confidence-weighted KL with penalty for high-entropy states"""
        #print("we're in confidence weighted KL")
        if len(beliefs) == 1:
            return beliefs[0].copy()
        
        # Equal weights for base KL
        weights = np.ones(len(beliefs)) / len(beliefs)
        
        def objective(merged_flat):
            merged = np.clip(merged_flat, 1e-10, 1)
            merged = merged / np.sum(merged)
            
            # Standard KL divergence term
            kl_term = 0
            for i, belief in enumerate(beliefs):
                belief_clip = np.clip(belief, 1e-10, 1)
                # kl = np.sum(belief_clip * np.log(belief_clip / merged))
                kl = np.sum(merged * np.log(merged / belief_clip))
                kl_term += weights[i] * kl
            
            # Confidence penalty term
            confidence_penalty = 0
            for j in range(self.n_states):
                if merged[j] > eta_threshold:
                    # Penalize disagreement on high-confidence states
                    variance = np.sum([(beliefs[i][j] - merged[j])**2 for i in range(len(beliefs))])
                    confidence_penalty += variance
            
            return kl_term + lambda_penalty * confidence_penalty
        
        # Initial guess
        initial = np.mean(beliefs, axis=0)
        initial = initial / np.sum(initial)
        
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

    def tune_confidence_hyperparameters(self, beliefs_sample: List[np.ndarray], 
                                   ground_truth_sample: np.ndarray) -> Tuple[float, float]:
        """Find optimal lambda and eta for confidence-weighted merge"""
        from scipy.optimize import differential_evolution
        
        def evaluate_params(params):
            lambda_val, eta_val = params
            merged = self.merge_beliefs_confidence_weighted(
                beliefs_sample, lambda_val, eta_val
            )
            # Compute KL from merged to ground truth
            gt_clip = np.clip(ground_truth_sample, 1e-10, 1)
            merged_clip = np.clip(merged, 1e-10, 1)
            kl = np.sum(gt_clip * np.log(gt_clip / merged_clip))
            return kl
        
        # Search bounds
        bounds = [(0.01, 1.0), (0.1, 0.5)]  # lambda, eta
        
        result = differential_evolution(
            evaluate_params,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )
        
        return result.x[0], result.x[1]  # optimal lambda, eta
    
    def run_trial(self, merge_interval: int, target_trajectory: List[int], trial_seed: int,
                  merge_method: str = 'standard_kl') -> Dict:
        """Run a single trial with KL method at specified merge interval"""
        #print(f"method used {merge_method}")
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
            'entropy_merged': [],
            'entropy_truth': [],
            'target_prob_merged': [],
            'target_prob_truth': [],
            'merge_events': [],
            'full_beliefs': [], 
            # NEW: Information tracking metrics
            'ground_truth_evolution': [],  # KL(truth[t] || truth[t-1])
            'cumulative_info_since_merge': [],  # Sum of evolution since last merge
            'steps_since_merge': []  # Steps since last communication
        }
        
        max_steps = min(len(target_trajectory), 1000)

        # Initialize visit counts for visit-weighted method
        visit_counts = np.zeros((self.n_agents, self.n_states))
        observation_counts = np.zeros((self.n_agents, self.n_states))  # Track positive observations
        recent_predictions = [[] for _ in range(self.n_agents)]  # Track recent accuracy
        # Hyperparameters for confidence-weighted
        optimal_lambda = 0.1  # Default
        optimal_eta = 0.3     # Default
        hyperparams_tuned = False
        # For tracking ground truth evolution
        previous_ground_truth = None
        last_merge_step = 0
        cumulative_info = 0.0

        for step in range(max_steps):
            target_pos = target_trajectory[step]
            
            # Simple agent movement (random walk)
            new_positions = []
            for i, pos in enumerate(positions):
                visit_counts[i, pos] += 1
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

                # Track positive observations for observation weighting
                if obs == 1:
                    observation_counts[i, pos] += 1
                
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

            ground_truth = self.compute_ground_truth_belief(all_observations)

            # Track ground truth evolution (how much did optimal belief change?)
            if previous_ground_truth is not None:
                # Compute KL divergence between current and previous ground truth
                prev_gt_clipped = np.clip(previous_ground_truth, 1e-10, 1)
                curr_gt_clipped = np.clip(ground_truth, 1e-10, 1)
                gt_evolution = np.sum(curr_gt_clipped * np.log(curr_gt_clipped / prev_gt_clipped))
                
                # Accumulate information since last merge
                cumulative_info += gt_evolution
            else:
                gt_evolution = 0.0
            
            # Store metrics
            metrics['ground_truth_evolution'].append(gt_evolution)
            metrics['cumulative_info_since_merge'].append(cumulative_info)
            metrics['steps_since_merge'].append(step - last_merge_step)
            
            # Update previous ground truth for next iteration
            previous_ground_truth = ground_truth.copy()

            # Compute ground truth
            current_merged = np.mean(beliefs, axis=0)

            # Determine current belief for metrics based on merge strategy
            if merge_interval == float('inf'):
                # Never merge - use first agent's belief
                current_belief = beliefs[0]
            elif merge_interval == 1:
                # Always merge - compute and apply merge every step
                current_belief = self.merge_beliefs_kl(beliefs)
                beliefs = [current_belief.copy() for _ in range(self.n_agents)]
            # Merge if scheduled
            elif merge_interval > 0 and step > 0 and step % merge_interval == 0:
                pre_merge_avg = np.mean(beliefs, axis=0)  # Average belief before merge

                # Choose merge method
                if merge_method == 'standard_kl':
                    merged = self.merge_beliefs_kl(beliefs)
                elif merge_method == 'visit_weighted':
                    merged = self.merge_beliefs_visit_weighted(beliefs, visit_counts)
                elif merge_method == 'observation_weighted':
                    merged = self.merge_beliefs_observation_weighted(beliefs, observation_counts)
                elif merge_method == 'entropy_weighted':
                    merged = self.merge_beliefs_entropy_weighted(beliefs)
                elif merge_method == 'confidence_weighted':
                    # Tune hyperparameters on first merge
                    if not hyperparams_tuned and step >= 20:  # Need some data first
                        optimal_lambda, optimal_eta = self.tune_confidence_hyperparameters(
                            beliefs, ground_truth
                        )
                        hyperparams_tuned = True
                    merged = self.merge_beliefs_confidence_weighted(
                        beliefs, optimal_lambda, optimal_eta
                    )
                else:
                    raise ValueError(f"Unknown merge method: {merge_method}")
                
                beliefs = [merged.copy() for _ in range(self.n_agents)]
                current_belief = merged
                
                # Record merge event
                metrics['merge_events'].append({
                    'step': step,
                    'kl_before': self._compute_kl_to_truth(pre_merge_avg, all_observations),
                    'kl_after': self._compute_kl_to_truth(merged, all_observations),
                    'improvement': self._compute_kl_to_truth(pre_merge_avg, all_observations) - 
                                self._compute_kl_to_truth(merged, all_observations)
                })
                # Reset information tracking after merge
                last_merge_step = step
                cumulative_info = 0.0
            else:
                # Between merges - use average of beliefs (cheap approximation)
                current_belief = np.mean(beliefs, axis=0)
            
            # Calculate metrics using current_belief
            gt_clipped = np.clip(ground_truth, 1e-10, 1)
            belief_clipped = np.clip(current_belief, 1e-10, 1)
            kl_div = np.sum(gt_clipped * np.log(gt_clipped / belief_clipped))
            
            metrics['kl_divergence_to_truth'].append(kl_div)
            metrics['entropy_merged'].append(entropy(current_belief))
            metrics['entropy_truth'].append(entropy(ground_truth))
            metrics['target_prob_merged'].append(current_belief[target_pos])
            metrics['target_prob_truth'].append(ground_truth[target_pos])
            
            metrics['full_beliefs'].append({
            'step': step,
            'target_position': target_pos,
            'individual_beliefs': [b.copy() for b in beliefs],
            'ground_truth': ground_truth.copy(),
            'agent_positions': positions.copy(),
            'observations': [obs_data]
            })
            
        # Final calculations
        predicted_target = np.argmax(current_merged)
        actual_target = target_trajectory[-1]
        
        # Prediction error (distance)
        pred_r, pred_c = divmod(predicted_target, self.cols)
        actual_r, actual_c = divmod(actual_target, self.cols)
        prediction_error = np.sqrt((pred_r - actual_r)**2 + (pred_c - actual_c)**2)
        
        return {
            'merge_interval': merge_interval,
            'merge_method': merge_method,
            'trial_seed': trial_seed,
            
            # Performance metrics
            'avg_kl_to_truth': np.mean(metrics['kl_divergence_to_truth']),
            'final_kl_to_truth': metrics['kl_divergence_to_truth'][-1],
            'avg_target_prob_merged': np.mean(metrics['target_prob_merged']),
            'avg_target_prob_truth': np.mean(metrics['target_prob_truth']),
            'final_target_prob_merged': metrics['target_prob_merged'][-1],
            'final_target_prob_truth': metrics['target_prob_truth'][-1],
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
        """Generate all trial tasks for all methods"""
        tasks = []
        
        # All methods to test
        merge_methods = self.config['merge_methods']
        
        for grid_size in self.config['grid_sizes']:
            for n_agents in self.config['n_agents_list']:
                for pattern in self.config['target_patterns']:
                    for trial_id in range(self.config['n_trials']):
                        trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                        
                        for merge_interval in self.config['merge_intervals']:
                            for merge_method in merge_methods:
                                task = {
                                    'grid_size': grid_size,
                                    'n_agents': n_agents,
                                    'pattern': pattern,
                                    'trial_id': trial_id,
                                    'merge_interval': merge_interval,
                                    'merge_method': merge_method,
                                    'trial_seed': trial_seed,
                                    'config': self.config
                                }
                                tasks.append(task)
        
        print(f"Generated {len(tasks)} total tasks ({len(merge_methods)} methods)")
        return tasks
    
    def get_checkpoint_path(self, task: Dict) -> str:
        """Get checkpoint file path for task"""
        grid_str = f"{task['grid_size'][0]}x{task['grid_size'][1]}"
        interval_str = 'inf' if task['merge_interval'] == float('inf') else str(task['merge_interval'])
        method_str = task.get('merge_method', 'standard_kl')  # Default to standard if not specified
        filename = (f"grid{grid_str}_agents{task['n_agents']}_{task['pattern']}_"
                f"trial{task['trial_id']}_interval{interval_str}_"
                f"{method_str}_seed{task['trial_seed']}.pkl")  # ADD method to filename
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
                        # Store by interval key, then by method
                        if merge_interval == 1:
                            interval_key = 'immediate_merge'
                        elif merge_interval == float('inf'):
                            interval_key = 'no_merge'
                        else:
                            interval_key = f'interval_{merge_interval}'
                        
                        # Initialize dict for both methods
                        pattern_results[interval_key] = {method: [] for method in self.config['merge_methods']}
                        
                        for trial_id in range(self.config['n_trials']):
                            trial_seed = generate_consistent_seed(grid_size, n_agents, pattern, trial_id)
                            
                            # Load all methods
                            for merge_method in self.config['merge_methods']:
                                task = {
                                    'grid_size': grid_size,
                                    'n_agents': n_agents,
                                    'pattern': pattern,
                                    'trial_id': trial_id,
                                    'merge_interval': merge_interval,
                                    'merge_method': merge_method,
                                    'trial_seed': trial_seed
                                }
                                
                                checkpoint_path = self.get_checkpoint_path(task)
                                
                                if os.path.exists(checkpoint_path):
                                    try:
                                        with open(checkpoint_path, 'rb') as f:
                                            result = pickle.load(f)
                                        pattern_results[interval_key][merge_method].append(result)
                                    except Exception as e:
                                        print(f"Failed to load {checkpoint_path}: {e}")
                    
                    all_results[grid_key][agent_key][pattern] = pattern_results
        
        # Save consolidated results - separate files for each method
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save combined file
        consolidated_path = self.results_dir / f"kl_all_methods_{timestamp}.pkl"
        with open(consolidated_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Combined results saved to {consolidated_path}")
        
        # Also save separate files for each method
        for method in self.config['merge_methods']:
            method_results = self._extract_method_results(all_results, method)
            method_path = self.results_dir / f"kl_{method}_{timestamp}.pkl"
            with open(method_path, 'wb') as f:
                pickle.dump(method_results, f)
            print(f"{method} results saved to {method_path}")
        
        return all_results
    
    def _extract_method_results(self, all_results: Dict, method: str) -> Dict:
        """Extract results for a single method"""
        #print(f"we are collecting result for {method} pkl")
        method_results = {}
        
        for grid_key, grid_data in all_results.items():
            method_results[grid_key] = {}
            
            for agent_key, agent_data in grid_data.items():
                method_results[grid_key][agent_key] = {}
                
                for pattern, pattern_data in agent_data.items():
                    method_results[grid_key][agent_key][pattern] = {}
                    
                    for interval_key, interval_data in pattern_data.items():
                        if method in interval_data:
                            # Store in format compatible with old visualization code
                            method_results[grid_key][agent_key][pattern][interval_key] = {
                                'kl_divergence': interval_data[method]
                            }
        
        return method_results
    
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
            task['merge_interval'], trajectory, task['trial_seed'], 
            task.get('merge_method', 'standard_kl')  # ADD method parameter
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
    method_str = task.get('merge_method', 'standard_kl')  # ADD THIS
    return (f"grid{grid_str}_agents{task['n_agents']}_{task['pattern']}_"
            f"trial{task['trial_id']}_interval{interval_str}_"
            f"{method_str}_seed{task['trial_seed']}.pkl")  # ADD method_str here


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

    config['merge_intervals'] = [
    float('inf') if x == 'inf' else x 
    for x in config['merge_intervals']
    ]
    
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