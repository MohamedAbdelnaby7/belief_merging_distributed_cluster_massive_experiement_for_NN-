"""
Comprehensive KL Performance Analyzer
Usage: python comprehensive_kl_analysis.py results/consolidated_results_20241219_143022.pkl

Creates extensive visualizations showing KL divergence performance across:
- All grid sizes
- All merge intervals  
- All agent numbers
- All combinations thereof
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveKLAnalyzer:
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self.load_results()
        self.df = self.flatten_to_dataframe()
        self.output_dir = self.results_file.parent / "comprehensive_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from: {self.results_file}")
        print(f"Output directory: {self.output_dir}")
        self.print_data_summary()
    
    def load_results(self):
        """Load results from pickle file"""
        try:
            with open(self.results_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load results file: {e}")
    
    def flatten_to_dataframe(self) -> pd.DataFrame:
        """Convert nested results to flat DataFrame - FIXED VERSION FOR ALL METHODS"""
        data_rows = []
        
        print("\n[DEBUG] Flattening results to DataFrame...")
        
        for grid_key, grid_results in self.results.items():
            for agent_key, agent_results in grid_results.items():
                try:
                    n_agents = int(agent_key.split('_')[0])
                except (ValueError, IndexError):
                    print(f"[WARNING] Could not parse agent number from {agent_key}")
                    continue
                    
                for pattern, pattern_results in agent_results.items():
                    for interval_key, interval_data in pattern_results.items():
                        
                        # Parse merge interval
                        if interval_key == 'immediate_merge' or interval_key == 'interval_1':
                            merge_interval = 1
                        elif interval_key == 'no_merge':
                            merge_interval = float('inf')
                        elif interval_key.startswith('interval_'):
                            try:
                                merge_interval = int(interval_key.split('_')[1])
                            except:
                                merge_interval = float('inf')
                        else:
                            # Try to extract number
                            try:
                                merge_interval = float(interval_key)
                            except:
                                merge_interval = float('inf')
                        
                        # Handle different data structures
                        # CASE 1: Dict with method names as keys (NEW FORMAT - 5 methods)
                        if isinstance(interval_data, dict):
                            for method_name, trials in interval_data.items():
                                if not isinstance(trials, list):
                                    continue
                                    
                                for trial in trials:
                                    if not isinstance(trial, dict):
                                        continue
                                    
                                    # Extract merge method (use trial's method or dict key)
                                    merge_method = trial.get('merge_method', method_name)
                                    
                                    # Parse grid size
                                    try:
                                        if 'x' in grid_key:
                                            rows, cols = map(int, grid_key.split('x'))
                                            grid_area = rows * cols
                                        else:
                                            grid_area = 100
                                            rows = cols = 10
                                    except:
                                        grid_area = 100
                                        rows = cols = 10
                                    
                                    data_rows.append({
                                        'grid_size': grid_key,
                                        'grid_area': grid_area,
                                        'grid_rows': rows,
                                        'grid_cols': cols,
                                        'n_agents': n_agents,
                                        'pattern': pattern,
                                        'interval_strategy': interval_key,
                                        'merge_interval': merge_interval,
                                        'merge_method': merge_method,
                                        'avg_kl_to_truth': trial.get('avg_kl_to_truth', np.nan),
                                        'final_kl_to_truth': trial.get('final_kl_to_truth', np.nan),
                                        'avg_target_prob_merged': trial.get('avg_target_prob_merged', np.nan),
                                        'avg_target_prob_truth': trial.get('avg_target_prob_truth', np.nan),
                                        'final_target_prob_merged': trial.get('final_target_prob_merged', np.nan),
                                        'final_target_prob_truth': trial.get('final_target_prob_truth', np.nan),
                                        'prediction_error': trial.get('prediction_error', np.nan),
                                        'avg_consensus': trial.get('avg_consensus', np.nan),
                                        'n_merges': trial.get('n_merges', 0),
                                        'communication_efficiency': trial.get('communication_efficiency', 0),
                                        'trial_data': trial  # Keep full trial for detailed analysis
                                    })
                        
                        # CASE 2: List of trials (OLD FORMAT - single method)
                        elif isinstance(interval_data, list):
                            for trial in interval_data:
                                if not isinstance(trial, dict):
                                    continue
                                
                                merge_method = trial.get('merge_method', 'unknown')
                                
                                try:
                                    if 'x' in grid_key:
                                        rows, cols = map(int, grid_key.split('x'))
                                        grid_area = rows * cols
                                    else:
                                        grid_area = 100
                                        rows = cols = 10
                                except:
                                    grid_area = 100
                                    rows = cols = 10
                                
                                data_rows.append({
                                    'grid_size': grid_key,
                                    'grid_area': grid_area,
                                    'grid_rows': rows,
                                    'grid_cols': cols,
                                    'n_agents': n_agents,
                                    'pattern': pattern,
                                    'interval_strategy': interval_key,
                                    'merge_interval': merge_interval,
                                    'merge_method': merge_method,
                                    'avg_kl_to_truth': trial.get('avg_kl_to_truth', np.nan),
                                    'final_kl_to_truth': trial.get('final_kl_to_truth', np.nan),
                                    'avg_target_prob_merged': trial.get('avg_target_prob_merged', np.nan),
                                    'avg_target_prob_truth': trial.get('avg_target_prob_truth', np.nan),
                                    'final_target_prob_merged': trial.get('final_target_prob_merged', np.nan),
                                    'final_target_prob_truth': trial.get('final_target_prob_truth', np.nan),
                                    'prediction_error': trial.get('prediction_error', np.nan),
                                    'avg_consensus': trial.get('avg_consensus', np.nan),
                                    'n_merges': trial.get('n_merges', 0),
                                    'communication_efficiency': trial.get('communication_efficiency', 0),
                                    'trial_data': trial
                                })
        
        if not data_rows:
            print("[ERROR] No valid data found in results file!")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_rows)
        df = df.dropna(subset=['avg_kl_to_truth'])
        
        # Print summary
        print(f"[DEBUG] Flattened {len(df)} trials")
        print(f"[DEBUG] Merge methods found: {sorted(df['merge_method'].unique())}")
        print(f"[DEBUG] Merge intervals: {sorted(df['merge_interval'].unique())}")
        print(f"[DEBUG] Grid sizes: {sorted(df['grid_size'].unique())}")
        print(f"[DEBUG] Agent numbers: {sorted(df['n_agents'].unique())}")
        
        # Count by method
        print(f"\n[DEBUG] Trials per method:")
        for method in sorted(df['merge_method'].unique()):
            count = len(df[df['merge_method'] == method])
            print(f"  {method}: {count} trials")
        
        return df
    
    def print_data_summary(self):
        """Print comprehensive data summary"""
        if self.df.empty:
            print("No data available")
            return
            
        print(f"\nDATA SUMMARY:")
        print(f"="*50)
        print(f"Total trials: {len(self.df):,}")
        print(f"Grid sizes: {sorted(self.df['grid_size'].unique())}")
        print(f"Grid areas: {sorted(self.df['grid_area'].unique())}")
        print(f"Agent numbers: {sorted(self.df['n_agents'].unique())}")
        
        # Fix the merge intervals display
        merge_intervals = self.df['merge_interval'].unique()
        finite_intervals = [x for x in merge_intervals if x != float('inf')]
        infinite_intervals = [x for x in merge_intervals if x == float('inf')]
        
        interval_display = sorted(finite_intervals)
        if infinite_intervals:
            interval_display.append('∞')
        
        print(f"Merge intervals: {interval_display}")
        
        # Show data distribution
        print(f"\nDATA DISTRIBUTION:")
        for grid in sorted(self.df['grid_size'].unique()):
            for agents in sorted(self.df['n_agents'].unique()):
                for interval in sorted(self.df['merge_interval'].unique()):
                    count = len(self.df[(self.df['grid_size'] == grid) & 
                                       (self.df['n_agents'] == agents) & 
                                       (self.df['merge_interval'] == interval)])
                    if count > 0:
                        int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                        print(f"  Grid {grid}, {agents} agents, interval {int_str}: {count} trials")
    
    def create_comprehensive_analysis(self):
        """Create the most comprehensive analysis possible"""
        
        # Get unique values for each dimension
        grid_sizes = sorted(self.df['grid_size'].unique())
        agent_numbers = sorted(self.df['n_agents'].unique())
        merge_intervals = sorted(self.df['merge_interval'].unique())
        
        n_grids = len(grid_sizes)
        n_agents = len(agent_numbers)
        n_intervals = len(merge_intervals)
        
        print(f"\nCreating comprehensive analysis:")
        print(f"  {n_grids} grid sizes × {n_agents} agent counts × {n_intervals} intervals")
        print(f"  Total combinations: {n_grids * n_agents * n_intervals}")
        
        # Create multiple large figures to show everything
        self.create_grid_size_analysis(grid_sizes, agent_numbers, merge_intervals)
        self.create_agent_count_analysis(grid_sizes, agent_numbers, merge_intervals)
        self.create_merge_interval_analysis(grid_sizes, agent_numbers, merge_intervals)
        self.create_three_way_interaction_analysis(grid_sizes, agent_numbers, merge_intervals)
        self.create_performance_surfaces()
        self.create_statistical_analysis()
        
    def create_grid_size_analysis(self, grid_sizes, agent_numbers, merge_intervals):
        """Comprehensive analysis by grid size"""
        
        # Create large figure with subplots for each grid size
        n_grids = len(grid_sizes)
        fig = plt.figure(figsize=(24, 6 * n_grids))
        
        for i, grid in enumerate(grid_sizes):
            grid_data = self.df[self.df['grid_size'] == grid]
            
            # Plot 1: Performance by merge interval for this grid size
            ax1 = plt.subplot(n_grids, 4, i*4 + 1)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(agent_numbers)))
            
            for j, agents in enumerate(agent_numbers):
                agent_grid_data = grid_data[grid_data['n_agents'] == agents]
                if not agent_grid_data.empty:
                    perf_by_interval = agent_grid_data.groupby('merge_interval')['avg_kl_to_truth'].agg(['mean', 'std', 'count']).reset_index()
                    
                    # Plot with error bars
                    x_vals = [list(merge_intervals).index(interval) for interval in perf_by_interval['merge_interval']]
                    ax1.errorbar(x_vals, perf_by_interval['mean'], yerr=perf_by_interval['std'],
                               color=colors[j], marker='o', linewidth=2, markersize=6,
                               label=f'{agents} agents', capsize=5)
                    
                    # Add sample size annotations
                    for x, mean_val, count in zip(x_vals, perf_by_interval['mean'], perf_by_interval['count']):
                        ax1.annotate(f'n={count}', (x, mean_val), xytext=(0, 10), 
                                   textcoords='offset points', ha='center', fontsize=8)
            
            ax1.set_xticks(range(len(merge_intervals)))
            ax1.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
            ax1.set_xlabel('Merge Interval')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            ax1.set_title(f'Grid {grid}: Performance by Merge Interval')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Box plots by agent number
            ax2 = plt.subplot(n_grids, 4, i*4 + 2)
            
            box_data = []
            labels = []
            colors_box = []
            
            for agents in agent_numbers:
                agent_data = grid_data[grid_data['n_agents'] == agents]['avg_kl_to_truth']
                if not agent_data.empty:
                    box_data.append(agent_data.values)
                    labels.append(f'{agents} agents')
                    colors_box.append(colors[agent_numbers.index(agents)])
            
            if box_data:
                bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax2.set_xlabel('Number of Agents')
            ax2.set_ylabel('KL Divergence to Ground Truth')
            ax2.set_title(f'Grid {grid}: Performance by Agent Count')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Heatmap of performance (agents × intervals)
            ax3 = plt.subplot(n_grids, 4, i*4 + 3)
            
            pivot_data = grid_data.groupby(['n_agents', 'merge_interval'])['avg_kl_to_truth'].mean().reset_index()
            if not pivot_data.empty:
                pivot_table = pivot_data.pivot(index='n_agents', columns='merge_interval', values='avg_kl_to_truth')
                
                # Rename columns for better display
                pivot_table.columns = [f"{int(col)}" if col != float('inf') else "∞" for col in pivot_table.columns]
                
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', ax=ax3, cbar_kws={'shrink': 0.8})
                ax3.set_title(f'Grid {grid}: Performance Heatmap')
                ax3.set_xlabel('Merge Interval')
                ax3.set_ylabel('Number of Agents')
            
            # Plot 4: Statistical summary
            ax4 = plt.subplot(n_grids, 4, i*4 + 4)
            ax4.axis('off')
            
            # Calculate statistics for this grid
            grid_stats = []
            for agents in agent_numbers:
                for interval in merge_intervals:
                    subset = grid_data[(grid_data['n_agents'] == agents) & (grid_data['merge_interval'] == interval)]
                    if not subset.empty:
                        grid_stats.append({
                            'agents': agents,
                            'interval': f"{int(interval)}" if interval != float('inf') else "∞",
                            'mean': subset['avg_kl_to_truth'].mean(),
                            'std': subset['avg_kl_to_truth'].std(),
                            'count': len(subset),
                            'min': subset['avg_kl_to_truth'].min(),
                            'max': subset['avg_kl_to_truth'].max()
                        })
            
            if grid_stats:
                stats_df = pd.DataFrame(grid_stats)
                
                # Find best configuration for this grid
                best_config = stats_df.loc[stats_df['mean'].idxmin()]
                
                summary_text = f"""GRID {grid} SUMMARY
                
                                Best Configuration:
                                • {best_config['agents']} agents, interval {best_config['interval']}
                                • KL Divergence: {best_config['mean']:.4f} ± {best_config['std']:.4f}
                                • Based on {best_config['count']} trials
                                                
                                Performance Range:
                                • Best: {stats_df['mean'].min():.4f}
                                • Worst: {stats_df['mean'].max():.4f}
                                • Improvement: {(stats_df['mean'].max() - stats_df['mean'].min()) / stats_df['mean'].max() * 100:.1f}%
                                                
                                Total Trials: {len(grid_data):,}
                                Configurations: {len(stats_df)}"""
                
                ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                        fontfamily='monospace', fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Comprehensive Analysis by Grid Size', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        grid_analysis_path = self.output_dir / 'grid_size_comprehensive_analysis.png'
        plt.savefig(grid_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grid size analysis saved: {grid_analysis_path}")
    
    def create_agent_count_analysis(self, grid_sizes, agent_numbers, merge_intervals):
        """Comprehensive analysis by agent count"""
        
        n_agents = len(agent_numbers)
        fig = plt.figure(figsize=(24, 6 * n_agents))
        
        for i, agents in enumerate(agent_numbers):
            agent_data = self.df[self.df['n_agents'] == agents]
            
            # Plot 1: Performance by merge interval for this agent count
            ax1 = plt.subplot(n_agents, 4, i*4 + 1)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
            
            for j, grid in enumerate(grid_sizes):
                grid_agent_data = agent_data[agent_data['grid_size'] == grid]
                if not grid_agent_data.empty:
                    perf_by_interval = grid_agent_data.groupby('merge_interval')['avg_kl_to_truth'].agg(['mean', 'std', 'count']).reset_index()
                    
                    x_vals = [list(merge_intervals).index(interval) for interval in perf_by_interval['merge_interval']]
                    ax1.errorbar(x_vals, perf_by_interval['mean'], yerr=perf_by_interval['std'],
                               color=colors[j], marker='o', linewidth=2, markersize=6,
                               label=f'Grid {grid}', capsize=5)
            
            ax1.set_xticks(range(len(merge_intervals)))
            ax1.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
            ax1.set_xlabel('Merge Interval')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            ax1.set_title(f'{agents} Agents: Performance by Merge Interval')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Performance scaling with grid size
            ax2 = plt.subplot(n_agents, 4, i*4 + 2)
            
            for j, interval in enumerate(merge_intervals):
                interval_data = agent_data[agent_data['merge_interval'] == interval]
                if not interval_data.empty:
                    perf_by_grid = interval_data.groupby('grid_area')['avg_kl_to_truth'].mean().reset_index()
                    
                    if not perf_by_grid.empty:
                        color = plt.cm.viridis(j / len(merge_intervals))
                        label = f"Interval {int(interval)}" if interval != float('inf') else "No Merge"
                        ax2.loglog(perf_by_grid['grid_area'], perf_by_grid['avg_kl_to_truth'], 
                                  'o-', color=color, label=label, linewidth=2, markersize=6)
            
            ax2.set_xlabel('Grid Area (log scale)')
            ax2.set_ylabel('KL Divergence (log scale)')
            ax2.set_title(f'{agents} Agents: Scaling with Problem Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Interval comparison heatmap
            ax3 = plt.subplot(n_agents, 4, i*4 + 3)
            
            pivot_data = agent_data.groupby(['grid_size', 'merge_interval'])['avg_kl_to_truth'].mean().reset_index()
            if not pivot_data.empty:
                pivot_table = pivot_data.pivot(index='grid_size', columns='merge_interval', values='avg_kl_to_truth')
                pivot_table.columns = [f"{int(col)}" if col != float('inf') else "∞" for col in pivot_table.columns]
                
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', ax=ax3, cbar_kws={'shrink': 0.8})
                ax3.set_title(f'{agents} Agents: Performance Heatmap')
                ax3.set_xlabel('Merge Interval')
                ax3.set_ylabel('Grid Size')
            
            # Plot 4: Efficiency analysis
            ax4 = plt.subplot(n_agents, 4, i*4 + 4)
            
            # Calculate communication efficiency for each grid size
            efficiency_data = []
            
            for grid in grid_sizes:
                grid_agent_data = agent_data[agent_data['grid_size'] == grid]
                
                # Get baseline (no merge) performance
                baseline_data = grid_agent_data[grid_agent_data['merge_interval'] == float('inf')]
                baseline_perf = baseline_data['avg_kl_to_truth'].mean() if not baseline_data.empty else np.nan
                
                for interval in merge_intervals:
                    if interval != float('inf'):
                        interval_data = grid_agent_data[grid_agent_data['merge_interval'] == interval]
                        if not interval_data.empty and not np.isnan(baseline_perf):
                            interval_perf = interval_data['avg_kl_to_truth'].mean()
                            avg_merges = interval_data['n_merges'].mean()
                            
                            if avg_merges > 0:
                                improvement_per_merge = (baseline_perf - interval_perf) / avg_merges
                                efficiency_data.append({
                                    'grid': grid,
                                    'interval': interval,
                                    'efficiency': improvement_per_merge
                                })
            
            if efficiency_data:
                eff_df = pd.DataFrame(efficiency_data)
                
                for grid in grid_sizes:
                    grid_eff = eff_df[eff_df['grid'] == grid]
                    if not grid_eff.empty:
                        color = colors[grid_sizes.index(grid)]
                        ax4.plot(grid_eff['interval'], grid_eff['efficiency'], 
                                'o-', color=color, label=f'Grid {grid}', linewidth=2, markersize=6)
                
                ax4.set_xlabel('Merge Interval')
                ax4.set_ylabel('KL Improvement per Merge')
                ax4.set_title(f'{agents} Agents: Communication Efficiency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.suptitle('Comprehensive Analysis by Agent Count', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        agent_analysis_path = self.output_dir / 'agent_count_comprehensive_analysis.png'
        plt.savefig(agent_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Agent count analysis saved: {agent_analysis_path}")
    
    def create_merge_interval_analysis(self, grid_sizes, agent_numbers, merge_intervals):
        """Comprehensive analysis by merge interval"""
        
        n_intervals = len(merge_intervals)
        fig = plt.figure(figsize=(24, 6 * n_intervals))
        
        for i, interval in enumerate(merge_intervals):
            interval_data = self.df[self.df['merge_interval'] == interval]
            
            # Plot 1: Performance by grid size and agent count
            ax1 = plt.subplot(n_intervals, 4, i*4 + 1)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(agent_numbers)))
            
            for j, agents in enumerate(agent_numbers):
                agent_interval_data = interval_data[interval_data['n_agents'] == agents]
                if not agent_interval_data.empty:
                    perf_by_grid = agent_interval_data.groupby('grid_area')['avg_kl_to_truth'].agg(['mean', 'std']).reset_index()
                    
                    ax1.errorbar(perf_by_grid['grid_area'], perf_by_grid['mean'], yerr=perf_by_grid['std'],
                               color=colors[j], marker='o', linewidth=2, markersize=6,
                               label=f'{agents} agents', capsize=5)
            
            ax1.set_xlabel('Grid Area')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            int_str = f"{int(interval)}" if interval != float('inf') else "∞"
            ax1.set_title(f'Interval {int_str}: Performance by Problem Size')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Agent scaling analysis
            ax2 = plt.subplot(n_intervals, 4, i*4 + 2)
            
            colors_grid = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
            
            for j, grid in enumerate(grid_sizes):
                grid_interval_data = interval_data[interval_data['grid_size'] == grid]
                if not grid_interval_data.empty:
                    perf_by_agents = grid_interval_data.groupby('n_agents')['avg_kl_to_truth'].agg(['mean', 'std']).reset_index()
                    
                    ax2.errorbar(perf_by_agents['n_agents'], perf_by_agents['mean'], yerr=perf_by_agents['std'],
                               color=colors_grid[j], marker='o', linewidth=2, markersize=6,
                               label=f'Grid {grid}', capsize=5)
            
            ax2.set_xlabel('Number of Agents')
            ax2.set_ylabel('KL Divergence to Ground Truth')
            ax2.set_title(f'Interval {int_str}: Performance by Team Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Performance distribution
            ax3 = plt.subplot(n_intervals, 4, i*4 + 3)
            
            # Create violin plots for each grid-agent combination
            violin_data = []
            violin_labels = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    subset = interval_data[(interval_data['grid_size'] == grid) & (interval_data['n_agents'] == agents)]
                    if not subset.empty and len(subset) > 2:  # Need at least 3 points for violin
                        violin_data.append(subset['avg_kl_to_truth'].values)
                        violin_labels.append(f'{grid}\n{agents}ag')
            
            if violin_data:
                parts = ax3.violinplot(violin_data, showmeans=True, showextrema=True)
                ax3.set_xticks(range(1, len(violin_labels) + 1))
                ax3.set_xticklabels(violin_labels, rotation=45)
                ax3.set_ylabel('KL Divergence to Ground Truth')
                ax3.set_title(f'Interval {int_str}: Performance Distribution')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Target tracking performance
            ax4 = plt.subplot(n_intervals, 4, i*4 + 4)
            
            # Scatter plot of target probability vs KL divergence
            colors_scatter = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
            
            for j, grid in enumerate(grid_sizes):
                grid_data = interval_data[interval_data['grid_size'] == grid]
                if not grid_data.empty:
                    ax4.scatter(grid_data['avg_target_prob_merged'], grid_data['avg_kl_to_truth'],
                              color=colors_scatter[j], alpha=0.6, s=50, label=f'Grid {grid}')
            
            ax4.set_xlabel('Target Probability (KL Method)')
            ax4.set_ylabel('KL Divergence to Ground Truth')
            ax4.set_title(f'Interval {int_str}: Target Tracking vs Information Quality')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Analysis by Merge Interval', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        interval_analysis_path = self.output_dir / 'merge_interval_comprehensive_analysis.png'
        plt.savefig(interval_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Merge interval analysis saved: {interval_analysis_path}")
    
    def create_three_way_interaction_analysis(self, grid_sizes, agent_numbers, merge_intervals):
        """Analyze three-way interactions between all factors"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 3D surface plot of performance
        ax1 = plt.subplot(2, 3, 1, projection='3d')
        
        # Create meshgrid for surface plot
        grid_areas = [self.df[self.df['grid_size'] == g]['grid_area'].iloc[0] for g in grid_sizes]
        
        for interval in merge_intervals:
            interval_data = self.df[self.df['merge_interval'] == interval]
            
            if not interval_data.empty:
                X, Y, Z = [], [], []
                
                for grid_area in grid_areas:
                    for agents in agent_numbers:
                        subset = interval_data[(interval_data['grid_area'] == grid_area) & 
                                             (interval_data['n_agents'] == agents)]
                        if not subset.empty:
                            X.append(grid_area)
                            Y.append(agents)
                            Z.append(subset['avg_kl_to_truth'].mean())
                
                if X:
                    color = plt.cm.viridis(list(merge_intervals).index(interval) / len(merge_intervals))
                    int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                    ax1.scatter(X, Y, Z, color=color, s=50, alpha=0.7, label=f'Int {int_str}')
        
        ax1.set_xlabel('Grid Area')
        ax1.set_ylabel('Number of Agents')
        ax1.set_zlabel('KL Divergence')
        ax1.set_title('3D Performance Landscape')
        ax1.legend()
        
        # 2. Grid size vs interval heatmap averaged across agents
        ax2 = plt.subplot(2, 3, 2)
        
        heatmap_data = self.df.groupby(['grid_size', 'merge_interval'])['avg_kl_to_truth'].mean().reset_index()
        if not heatmap_data.empty:
            pivot_table = heatmap_data.pivot(index='grid_size', columns='merge_interval', values='avg_kl_to_truth')
            pivot_table.columns = [f"{int(col)}" if col != float('inf') else "∞" for col in pivot_table.columns]
            
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', ax=ax2)
            ax2.set_title('Grid Size × Merge Interval\n(Averaged across agents)')
            ax2.set_xlabel('Merge Interval')
            ax2.set_ylabel('Grid Size')
        
        # 3. Agent count vs interval heatmap averaged across grids
        ax3 = plt.subplot(2, 3, 3)
        
        heatmap_data2 = self.df.groupby(['n_agents', 'merge_interval'])['avg_kl_to_truth'].mean().reset_index()
        if not heatmap_data2.empty:
            pivot_table2 = heatmap_data2.pivot(index='n_agents', columns='merge_interval', values='avg_kl_to_truth')
            pivot_table2.columns = [f"{int(col)}" if col != float('inf') else "∞" for col in pivot_table2.columns]
            
            sns.heatmap(pivot_table2, annot=True, fmt='.3f', cmap='viridis_r', ax=ax3)
            ax3.set_title('Agent Count × Merge Interval\n(Averaged across grids)')
            ax3.set_xlabel('Merge Interval')
            ax3.set_ylabel('Number of Agents')
        
        # 4. Grid size vs agent count heatmap averaged across intervals
        ax4 = plt.subplot(2, 3, 4)
        
        heatmap_data3 = self.df.groupby(['grid_size', 'n_agents'])['avg_kl_to_truth'].mean().reset_index()
        if not heatmap_data3.empty:
            pivot_table3 = heatmap_data3.pivot(index='grid_size', columns='n_agents', values='avg_kl_to_truth')
            
            sns.heatmap(pivot_table3, annot=True, fmt='.3f', cmap='viridis_r', ax=ax4)
            ax4.set_title('Grid Size × Agent Count\n(Averaged across intervals)')
            ax4.set_xlabel('Number of Agents')
            ax4.set_ylabel('Grid Size')
        
        # 5. Best configuration for each grid size
        ax5 = plt.subplot(2, 3, 5)
        
        best_configs = []
        for grid in grid_sizes:
            grid_data = self.df[self.df['grid_size'] == grid]
            best_idx = grid_data['avg_kl_to_truth'].idxmin()
            best_config = grid_data.loc[best_idx]
            best_configs.append({
                'grid': grid,
                'best_agents': best_config['n_agents'],
                'best_interval': best_config['merge_interval'],
                'best_performance': best_config['avg_kl_to_truth']
            })
        
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            
            # Plot best agent count for each grid
            ax5.scatter(range(len(grid_sizes)), best_df['best_agents'], 
                       c=best_df['best_performance'], cmap='viridis_r', s=100, alpha=0.8)
            
            ax5.set_xticks(range(len(grid_sizes)))
            ax5.set_xticklabels(grid_sizes, rotation=45)
            ax5.set_xlabel('Grid Size')
            ax5.set_ylabel('Optimal Number of Agents')
            ax5.set_title('Optimal Agent Count by Grid Size')
            
            # Add text annotations
            for i, (grid, agents, interval, perf) in enumerate(zip(best_df['grid'], best_df['best_agents'], 
                                                                 best_df['best_interval'], best_df['best_performance'])):
                int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                ax5.annotate(f'Int {int_str}\n{perf:.3f}', (i, agents), 
                           xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
        
        # 6. Performance improvement matrix
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate improvement from worst to best for each grid-agent combination
        improvement_data = []
        
        for grid in grid_sizes:
            for agents in agent_numbers:
                subset = self.df[(self.df['grid_size'] == grid) & (self.df['n_agents'] == agents)]
                if not subset.empty and len(subset) > 1:
                    worst_perf = subset['avg_kl_to_truth'].max()
                    best_perf = subset['avg_kl_to_truth'].min()
                    improvement = (worst_perf - best_perf) / worst_perf * 100
                    
                    improvement_data.append({
                        'grid': grid,
                        'agents': agents,
                        'improvement': improvement
                    })
        
        if improvement_data:
            imp_df = pd.DataFrame(improvement_data)
            pivot_imp = imp_df.pivot(index='grid', columns='agents', values='improvement')
            
            sns.heatmap(pivot_imp, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax6, 
                       cbar_kws={'label': 'Improvement %'})
            ax6.set_title('Max Improvement Potential\n(Best vs Worst Interval)')
            ax6.set_xlabel('Number of Agents')
            ax6.set_ylabel('Grid Size')
        
        plt.suptitle('Three-Way Interaction Analysis: Grid × Agents × Intervals', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        interaction_path = self.output_dir / 'three_way_interaction_analysis.png'
        plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Three-way interaction analysis saved: {interaction_path}")
    
    def create_performance_surfaces(self):
        """Create 3D surface plots for performance analysis"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create separate surface for each merge interval
        n_intervals = len(self.df['merge_interval'].unique())
        n_cols = 3
        n_rows = (n_intervals + n_cols - 1) // n_cols
        
        merge_intervals = sorted(self.df['merge_interval'].unique())
        
        for i, interval in enumerate(merge_intervals):
            ax = plt.subplot(n_rows, n_cols, i+1, projection='3d')
            
            interval_data = self.df[self.df['merge_interval'] == interval]
            
            if not interval_data.empty:
                # Create meshgrid
                grid_areas = sorted(interval_data['grid_area'].unique())
                agent_counts = sorted(interval_data['n_agents'].unique())
                
                X, Y, Z = [], [], []
                
                for grid_area in grid_areas:
                    for agents in agent_counts:
                        subset = interval_data[(interval_data['grid_area'] == grid_area) & 
                                             (interval_data['n_agents'] == agents)]
                        if not subset.empty:
                            X.append(grid_area)
                            Y.append(agents)
                            Z.append(subset['avg_kl_to_truth'].mean())
                
                if len(X) > 3:  # Need enough points for surface
                    # Convert to arrays and create surface
                    X = np.array(X)
                    Y = np.array(Y)
                    Z = np.array(Z)
                    
                    # Create surface plot
                    ax.scatter(X, Y, Z, c=Z, cmap='viridis_r', s=50, alpha=0.8)
                    
                    # Try to create a surface if we have enough regular data
                    if len(grid_areas) > 1 and len(agent_counts) > 1:
                        try:
                            from scipy.interpolate import griddata
                            
                            # Create regular grid
                            xi = np.linspace(min(grid_areas), max(grid_areas), 10)
                            yi = np.linspace(min(agent_counts), max(agent_counts), 10)
                            XI, YI = np.meshgrid(xi, yi)
                            
                            # Interpolate
                            ZI = griddata((X, Y), Z, (XI, YI), method='linear')
                            
                            # Plot surface
                            ax.plot_surface(XI, YI, ZI, alpha=0.3, cmap='viridis_r')
                        except:
                            pass  # Skip surface if interpolation fails
            
            int_str = f"Interval {int(interval)}" if interval != float('inf') else "No Merge"
            ax.set_title(int_str)
            ax.set_xlabel('Grid Area')
            ax.set_ylabel('Agents')
            ax.set_zlabel('KL Divergence')
        
        plt.suptitle('Performance Surfaces by Merge Interval', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        surfaces_path = self.output_dir / 'performance_surfaces.png'
        plt.savefig(surfaces_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance surfaces saved: {surfaces_path}")
    
    def create_statistical_analysis(self):
        """Comprehensive statistical analysis"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. ANOVA-style analysis
        ax1 = plt.subplot(3, 3, 1)
        
        # Effect size of each factor
        from scipy import stats
        
        # Calculate effect sizes
        effects = []
        
        # Grid size effect
        grid_means = self.df.groupby('grid_size')['avg_kl_to_truth'].mean()
        grid_var = np.var(grid_means)
        total_var = np.var(self.df['avg_kl_to_truth'])
        grid_effect = grid_var / total_var if total_var > 0 else 0
        effects.append(('Grid Size', grid_effect))
        
        # Agent count effect
        agent_means = self.df.groupby('n_agents')['avg_kl_to_truth'].mean()
        agent_var = np.var(agent_means)
        agent_effect = agent_var / total_var if total_var > 0 else 0
        effects.append(('Agent Count', agent_effect))
        
        # Merge interval effect
        interval_means = self.df.groupby('merge_interval')['avg_kl_to_truth'].mean()
        interval_var = np.var(interval_means)
        interval_effect = interval_var / total_var if total_var > 0 else 0
        effects.append(('Merge Interval', interval_effect))
        
        # Plot effect sizes
        if effects:
            factors, effect_sizes = zip(*effects)
            bars = ax1.bar(factors, effect_sizes, color=['blue', 'green', 'red'], alpha=0.7)
            ax1.set_ylabel('Effect Size (Variance Explained)')
            ax1.set_title('Factor Importance Analysis')
            ax1.set_ylim(0, 1)
            
            # Add value labels
            for bar, size in zip(bars, effect_sizes):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{size:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Pairwise comparisons for merge intervals
        ax2 = plt.subplot(3, 3, 2)
        
        merge_intervals = sorted(self.df['merge_interval'].unique())
        n_intervals = len(merge_intervals)
        p_matrix = np.ones((n_intervals, n_intervals))
        
        for i, int1 in enumerate(merge_intervals):
            for j, int2 in enumerate(merge_intervals):
                if i != j:
                    data1 = self.df[self.df['merge_interval'] == int1]['avg_kl_to_truth']
                    data2 = self.df[self.df['merge_interval'] == int2]['avg_kl_to_truth']
                    
                    if len(data1) > 4 and len(data2) > 4:
                        try:
                            _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                            p_matrix[i, j] = p_val
                        except:
                            pass
        
        im = ax2.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        ax2.set_xticks(range(n_intervals))
        ax2.set_yticks(range(n_intervals))
        ax2.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        ax2.set_yticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        ax2.set_title('Statistical Significance Matrix\n(p-values, green=significant)')
        
        # Add text annotations
        for i in range(n_intervals):
            for j in range(n_intervals):
                if i != j:
                    text = f"{p_matrix[i, j]:.3f}"
                    color = 'white' if p_matrix[i, j] < 0.05 else 'black'
                    ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax2, shrink=0.6)
        
        # 3. Performance ranking by configuration
        ax3 = plt.subplot(3, 3, 3)
        
        # Get top 10 and bottom 10 configurations
        config_performance = self.df.groupby(['grid_size', 'n_agents', 'merge_interval'])['avg_kl_to_truth'].agg(['mean', 'count']).reset_index()
        config_performance = config_performance[config_performance['count'] >= 3]  # At least 3 trials
        
        top_configs = config_performance.nsmallest(5, 'mean')
        bottom_configs = config_performance.nlargest(5, 'mean')
        
        # Plot as horizontal bar chart
        y_pos = range(len(top_configs) + len(bottom_configs))
        
        performances = list(top_configs['mean']) + list(bottom_configs['mean'])
        labels = []
        
        for _, row in top_configs.iterrows():
            int_str = f"{int(row['merge_interval'])}" if row['merge_interval'] != float('inf') else "∞"
            labels.append(f"{row['grid_size']}, {row['n_agents']}ag, int{int_str}")
        
        for _, row in bottom_configs.iterrows():
            int_str = f"{int(row['merge_interval'])}" if row['merge_interval'] != float('inf') else "∞"
            labels.append(f"{row['grid_size']}, {row['n_agents']}ag, int{int_str}")
        
        colors = ['green'] * len(top_configs) + ['red'] * len(bottom_configs)
        
        bars = ax3.barh(y_pos, performances, color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel('KL Divergence to Ground Truth')
        ax3.set_title('Best and Worst Configurations')
        ax3.axvline(x=np.median(self.df['avg_kl_to_truth']), color='black', linestyle='--', alpha=0.5, label='Median')
        
        # Add value labels
        for bar, perf in zip(bars, performances):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{perf:.3f}', va='center', fontsize=8)
        
        # 4. Correlation analysis
        ax4 = plt.subplot(3, 3, 4)
        
        # Create numerical versions of categorical variables
        df_numeric = self.df.copy()
        df_numeric['grid_area_log'] = np.log(df_numeric['grid_area'])
        df_numeric['merge_interval_inv'] = df_numeric['merge_interval'].replace(float('inf'), 0)
        df_numeric['merge_interval_inv'] = 1 / (df_numeric['merge_interval_inv'] + 1)  # Communication frequency
        
        # Calculate correlations
        corr_vars = ['grid_area_log', 'n_agents', 'merge_interval_inv', 'avg_kl_to_truth', 
                    'avg_target_prob_merged', 'prediction_error', 'n_merges']
        
        corr_matrix = df_numeric[corr_vars].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Correlation Matrix')
        
        # 5. Residual analysis (simplified without sklearn)
        ax5 = plt.subplot(3, 3, 5)
        
        # Simple correlation analysis instead of linear regression
        X = df_numeric[['grid_area_log', 'n_agents', 'merge_interval_inv']].values
        y = df_numeric['avg_kl_to_truth'].values
        
        # Calculate correlations manually
        correlations = []
        feature_names = ['Log Grid Area', 'N Agents', 'Comm Frequency']
        
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(corr)
        
        # Plot correlation coefficients
        bars = ax5.bar(feature_names, correlations, color=['blue', 'green', 'red'], alpha=0.7)
        ax5.set_ylabel('Correlation with KL Divergence')
        ax5.set_title('Feature Correlations')
        ax5.tick_params(axis='x', rotation=45)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                    f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # 6. Feature importance (simplified)
        ax6 = plt.subplot(3, 3, 6)
        
        # Use absolute correlations as feature importance
        importances = np.abs(correlations)
        
        bars = ax6.bar(feature_names, importances, color=['blue', 'green', 'red'], alpha=0.7)
        ax6.set_ylabel('Absolute Correlation')
        ax6.set_title('Feature Importance (Correlation-based)')
        ax6.tick_params(axis='x', rotation=45)
        
        # Add R² equivalent (max correlation squared)
        max_corr_r2 = max([corr**2 for corr in correlations])
        ax6.text(0.7, 0.9, f'Max R² ≈ {max_corr_r2:.3f}', transform=ax6.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. Interaction effects
        ax7 = plt.subplot(3, 3, 7)
        
        # Grid size × Agent count interaction
        interaction_data = []
        
        for grid in sorted(self.df['grid_size'].unique()):
            for agents in sorted(self.df['n_agents'].unique()):
                subset = self.df[(self.df['grid_size'] == grid) & (self.df['n_agents'] == agents)]
                if not subset.empty:
                    interaction_data.append({
                        'grid': grid,
                        'agents': agents,
                        'performance': subset['avg_kl_to_truth'].mean(),
                        'n_trials': len(subset)
                    })
        
        if interaction_data:
            int_df = pd.DataFrame(interaction_data)
            
            for grid in sorted(int_df['grid'].unique()):
                grid_data = int_df[int_df['grid'] == grid]
                ax7.plot(grid_data['agents'], grid_data['performance'], 
                        'o-', label=f'Grid {grid}', linewidth=2, markersize=6)
            
            ax7.set_xlabel('Number of Agents')
            ax7.set_ylabel('Mean KL Divergence')
            ax7.set_title('Grid × Agent Interaction')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Scaling analysis
        ax8 = plt.subplot(3, 3, 8)
        
        # Performance vs problem complexity (grid_area * n_agents)
        self.df['complexity'] = self.df['grid_area'] * self.df['n_agents']
        
        for interval in sorted(self.df['merge_interval'].unique()):
            interval_data = self.df[self.df['merge_interval'] == interval]
            if not interval_data.empty:
                complexity_perf = interval_data.groupby('complexity')['avg_kl_to_truth'].mean().reset_index()
                
                color = plt.cm.viridis(list(sorted(self.df['merge_interval'].unique())).index(interval) / len(self.df['merge_interval'].unique()))
                int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                
                ax8.loglog(complexity_perf['complexity'], complexity_perf['avg_kl_to_truth'], 
                          'o-', color=color, label=f'Int {int_str}', linewidth=2, markersize=6)
        
        ax8.set_xlabel('Problem Complexity (Grid Area × Agents)')
        ax8.set_ylabel('KL Divergence')
        ax8.set_title('Scaling with Problem Complexity')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary statistics table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate comprehensive statistics
        stats_text = f"""STATISTICAL SUMMARY
{'='*40}

Dataset:
  • Total trials: {len(self.df):,}
  • Configurations: {len(self.df.groupby(['grid_size', 'n_agents', 'merge_interval']))}
  • KL range: {self.df['avg_kl_to_truth'].min():.4f} - {self.df['avg_kl_to_truth'].max():.4f}

Factor Effects (Variance Explained):
  • Grid Size: {grid_effect:.3f}
  • Agent Count: {agent_effect:.3f}  
  • Merge Interval: {interval_effect:.3f}

Best Configuration:
  • {top_configs.iloc[0]['grid_size']}, {top_configs.iloc[0]['n_agents']} agents
  • Interval {int(top_configs.iloc[0]['merge_interval']) if top_configs.iloc[0]['merge_interval'] != float('inf') else '∞'}
  • KL Divergence: {top_configs.iloc[0]['mean']:.4f}

Correlation Analysis:
  • Grid Size: {correlations[0]:+.3f}
  • Agent Count: {correlations[1]:+.3f}
  • Communication Freq: {correlations[2]:+.3f}

Key Insights:
  • {'Grid size' if grid_effect == max(grid_effect, agent_effect, interval_effect) else 'Agent count' if agent_effect == max(grid_effect, agent_effect, interval_effect) else 'Merge interval'} has strongest effect
  • {'Positive' if correlations[2] > 0 else 'Negative'} correlation with communication frequency
  • Max explained variance: {max(grid_effect, agent_effect, interval_effect):.3f}"""
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        plt.suptitle('Comprehensive Statistical Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        stats_path = self.output_dir / 'statistical_analysis.png'
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical analysis saved: {stats_path}")
    def extract_time_series_data(self):
        """Extract and organize time series data from all trials"""
        time_series_data = {}
        
        for _, row in self.df.iterrows():
            grid = row['grid_size']
            #print(f"grid_size is {grid}")
            agents = row['n_agents']
            #print(f"n_agents is {agents}")
            interval = row['merge_interval']
            #print(f"interval is {interval}")
            
            # Create nested structure: grid -> agents -> interval -> series list
            if grid not in time_series_data:
                time_series_data[grid] = {}
                #print(f"grid is not in time_series_data")
                 
            if agents not in time_series_data[grid]:
                time_series_data[grid][agents] = {}
                #print(f"agents is not in time_series_data")
            if interval not in time_series_data[grid][agents]:
                #print(f"interval is not in time_series_data")
                time_series_data[grid][agents][interval] = {
                    'kl_series': [],
                    'target_prob_series': [],
                    'truth_prob_series': [],
                    'consensus_series': []
                }
            
            trial = row['trial_data']
            if 'time_series' in trial:
                ts = trial['time_series']
                if ts.get('kl_divergence_to_truth'):
                    time_series_data[grid][agents][interval]['kl_series'].append(ts['kl_divergence_to_truth'])
                if ts.get('target_prob_merged'):
                    time_series_data[grid][agents][interval]['target_prob_series'].append(ts['target_prob_merged'])
                if ts.get('target_prob_truth'):
                    time_series_data[grid][agents][interval]['truth_prob_series'].append(ts['target_prob_truth'])
                if ts.get('belief_consensus'):
                    time_series_data[grid][agents][interval]['consensus_series'].append(ts['belief_consensus'])
        
        return time_series_data
    
    def create_kl_evolution_analysis(self):
        """Create comprehensive KL divergence evolution analysis over time steps"""
        
        print("Creating KL evolution analysis...")
        
        time_series_data = self.extract_time_series_data()
        
        if not time_series_data:
            print("No time series data available for evolution analysis")
            return
        
        grid_sizes = sorted(time_series_data.keys())
        agent_numbers = sorted(set(agents for grid_data in time_series_data.values() 
                                 for agents in grid_data.keys()))
        merge_intervals = sorted(set(interval for grid_data in time_series_data.values() 
                                   for agent_data in grid_data.values() 
                                   for interval in agent_data.keys()))
        print(f"grids are {grid_sizes}, agents {agent_numbers}, intervals {merge_intervals}")
        
        
        # Create multiple figures for different perspectives
        
        # 1. Evolution by Merge Interval (main analysis)
        self.create_evolution_by_interval(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        
        # 2. Evolution by Grid Size
        self.create_evolution_by_grid_size(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        
        # 3. Evolution by Agent Count
        self.create_evolution_by_agent_count(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        
        # 4. Combined Evolution Overview
        self.create_evolution_overview(time_series_data, grid_sizes, agent_numbers, merge_intervals)
    
    def create_evolution_by_interval(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Show KL evolution for each merge interval across all conditions"""
        
        n_intervals = len(merge_intervals)
        fig = plt.figure(figsize=(24, 6 * n_intervals))
        
        colors_grid = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
        colors_agent = plt.cm.plasma(np.linspace(0, 1, len(agent_numbers)))
        
        for i, interval in enumerate(merge_intervals):
            
            # Plot 1: Average evolution across all configurations
            ax1 = plt.subplot(n_intervals, 4, i*4 + 1)
            
            # Collect all series for this interval
            all_series = []
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        all_series.extend(series_list)
            
            if all_series:
                # Pad and average all series
                max_len = max(len(series) for series in all_series)
                padded_series = []
                
                for series in all_series:
                    if len(series) < max_len:
                        padded = list(series) + [series[-1]] * (max_len - len(series))
                    else:
                        padded = series[:max_len]
                    padded_series.append(padded)
                
                mean_series = np.mean(padded_series, axis=0)
                std_series = np.std(padded_series, axis=0)
                steps = range(len(mean_series))
                
                ax1.plot(steps, mean_series, 'b-', linewidth=3, label=f'Mean (n={len(all_series)})')
                ax1.fill_between(steps, mean_series - std_series, mean_series + std_series,
                                alpha=0.3, color='blue')
                
                # Mark merge events
                if interval != float('inf') and interval > 0:
                    merge_points = range(int(interval), len(steps), int(interval))
                    for mp in merge_points[:5]:
                        ax1.axvline(mp, color='red', alpha=0.5, linestyle='--', linewidth=1)
                
                # Calculate and show trend
                if len(mean_series) > 10:
                    # Linear trend in second half
                    half_point = len(mean_series) // 2
                    y_trend = mean_series[half_point:]
                    x_trend = np.array(range(len(y_trend)))
                    if len(y_trend) > 1:
                        slope, intercept = np.polyfit(x_trend, y_trend, 1)
                        trend_line = slope * x_trend + intercept
                        ax1.plot(range(half_point, len(mean_series)), trend_line, 
                                'r--', linewidth=2, alpha=0.8, label=f'Trend: {slope:+.4f}/step')
            
            int_str = f"Interval {int(interval)}" if interval != float('inf') else "No Merge"
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            ax1.set_title(f'{int_str}: Overall Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Evolution by grid size
            ax2 = plt.subplot(n_intervals, 4, i*4 + 2)
            
            for j, grid in enumerate(grid_sizes):
                grid_series = []
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        grid_series.extend(series_list)
                
                if grid_series:
                    max_len = max(len(series) for series in grid_series)
                    padded_series = []
                    
                    for series in grid_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    ax2.plot(steps, mean_series, color=colors_grid[j], linewidth=2, 
                            label=f'Grid {grid} (n={len(grid_series)})', alpha=0.8)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('KL Divergence to Ground Truth')
            ax2.set_title(f'{int_str}: Evolution by Grid Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Evolution by agent count
            ax3 = plt.subplot(n_intervals, 4, i*4 + 3)
            
            for j, agents in enumerate(agent_numbers):
                agent_series = []
                for grid in grid_sizes:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        agent_series.extend(series_list)
                
                if agent_series:
                    max_len = max(len(series) for series in agent_series)
                    padded_series = []
                    
                    for series in agent_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    ax3.plot(steps, mean_series, color=colors_agent[j], linewidth=2, 
                            label=f'{agents} agents (n={len(agent_series)})', alpha=0.8)
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('KL Divergence to Ground Truth')
            ax3.set_title(f'{int_str}: Evolution by Agent Count')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Individual examples and variance
            ax4 = plt.subplot(n_intervals, 4, i*4 + 4)
            
            # Show a few individual trial examples
            example_count = 0
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents] and
                        example_count < 5):  # Show max 5 examples
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        if series_list:
                            series = series_list[0]  # Take first trial as example
                            ax4.plot(series, alpha=0.3, linewidth=1, color='gray')
                            example_count += 1
            
            # Overlay the mean
            if all_series:
                ax4.plot(steps, mean_series, 'b-', linewidth=3, label='Mean', alpha=0.9)
                
                # Show variance evolution
                variance_series = np.var(padded_series, axis=0)
                ax4_twin = ax4.twinx()
                ax4_twin.plot(steps, variance_series, 'orange', linewidth=2, alpha=0.7, label='Variance')
                ax4_twin.set_ylabel('Variance in KL Divergence', color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')
            
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('KL Divergence to Ground Truth')
            ax4.set_title(f'{int_str}: Individual Trials & Variance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('KL Divergence Evolution by Merge Interval', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        evolution_interval_path = self.output_dir / 'kl_evolution_by_interval.png'
        plt.savefig(evolution_interval_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"KL evolution by interval saved: {evolution_interval_path}")
    
    def create_evolution_by_grid_size(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Show KL evolution for each grid size across intervals and agents"""
        
        n_grids = len(grid_sizes)
        fig = plt.figure(figsize=(24, 6 * n_grids))
        
        colors_interval = plt.cm.viridis(np.linspace(0, 1, len(merge_intervals)))
        colors_agent = plt.cm.plasma(np.linspace(0, 1, len(agent_numbers)))
        
        for i, grid in enumerate(grid_sizes):
            
            # Plot 1: All intervals for this grid size
            ax1 = plt.subplot(n_grids, 4, i*4 + 1)
            
            for j, interval in enumerate(merge_intervals):
                interval_series = []
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        interval_series.extend(series_list)
                
                if interval_series:
                    max_len = max(len(series) for series in interval_series)
                    padded_series = []
                    
                    for series in interval_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    int_str = f"Int {int(interval)}" if interval != float('inf') else "No Merge"
                    ax1.plot(steps, mean_series, color=colors_interval[j], linewidth=2, 
                            label=f'{int_str} (n={len(interval_series)})', alpha=0.8)
                    
                    # Mark merge events
                    if interval != float('inf') and interval > 0:
                        merge_points = range(int(interval), len(steps), int(interval))
                        for mp in merge_points[:3]:
                            ax1.axvline(mp, color=colors_interval[j], alpha=0.3, linestyle=':', linewidth=1)
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            ax1.set_title(f'Grid {grid}: Evolution by Merge Interval')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: All agent counts for this grid size
            ax2 = plt.subplot(n_grids, 4, i*4 + 2)
            
            for j, agents in enumerate(agent_numbers):
                agent_series = []
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        agent_series.extend(series_list)
                
                if agent_series:
                    max_len = max(len(series) for series in agent_series)
                    padded_series = []
                    
                    for series in agent_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    ax2.plot(steps, mean_series, color=colors_agent[j], linewidth=2, 
                            label=f'{agents} agents (n={len(agent_series)})', alpha=0.8)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('KL Divergence to Ground Truth')
            ax2.set_title(f'Grid {grid}: Evolution by Agent Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Best vs worst configurations
            ax3 = plt.subplot(n_grids, 4, i*4 + 3)
            
            # Find best and worst performing configurations for this grid
            grid_configs = []
            for agents in agent_numbers:
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        if series_list:
                            avg_performance = np.mean([np.mean(series) for series in series_list])
                            grid_configs.append({
                                'agents': agents,
                                'interval': interval,
                                'performance': avg_performance,
                                'series_list': series_list
                            })
            
            if grid_configs:
                # Sort by performance
                grid_configs.sort(key=lambda x: x['performance'])
                
                # Plot best configuration
                best_config = grid_configs[0]
                best_series = best_config['series_list']
                if best_series:
                    max_len = max(len(series) for series in best_series)
                    padded_series = []
                    for series in best_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_best = np.mean(padded_series, axis=0)
                    steps = range(len(mean_best))
                    
                    int_str = f"{int(best_config['interval'])}" if best_config['interval'] != float('inf') else "∞"
                    ax3.plot(steps, mean_best, 'g-', linewidth=3, 
                            label=f'Best: {best_config["agents"]}ag, int{int_str}', alpha=0.8)
                
                # Plot worst configuration
                worst_config = grid_configs[-1]
                worst_series = worst_config['series_list']
                if worst_series:
                    max_len = max(len(series) for series in worst_series)
                    padded_series = []
                    for series in worst_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_worst = np.mean(padded_series, axis=0)
                    steps = range(len(mean_worst))
                    
                    int_str = f"{int(worst_config['interval'])}" if worst_config['interval'] != float('inf') else "∞"
                    ax3.plot(steps, mean_worst, 'r-', linewidth=3, 
                            label=f'Worst: {worst_config["agents"]}ag, int{int_str}', alpha=0.8)
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('KL Divergence to Ground Truth')
            ax3.set_title(f'Grid {grid}: Best vs Worst Configuration')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Convergence analysis
            ax4 = plt.subplot(n_grids, 4, i*4 + 4)
            
            # Calculate convergence rates for each configuration
            convergence_data = []
            
            for agents in agent_numbers:
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        slopes = []
                        for series in series_list:
                            if len(series) > 10:
                                # Calculate slope in second half
                                half_point = len(series) // 2
                                y = series[half_point:]
                                x = range(len(y))
                                if len(y) > 1:
                                    slope, _ = np.polyfit(x, y, 1)
                                    slopes.append(slope)
                        
                        if slopes:
                            mean_slope = np.mean(slopes)
                            int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                            convergence_data.append({
                                'config': f'{agents}ag-int{int_str}',
                                'agents': agents,
                                'interval': interval,
                                'slope': mean_slope,
                                'n_trials': len(slopes)
                            })
            
            if convergence_data:
                conv_df = pd.DataFrame(convergence_data)
                
                # Create bar plot of convergence rates
                x_pos = range(len(conv_df))
                colors = ['green' if slope < 0 else 'red' for slope in conv_df['slope']]
                
                bars = ax4.bar(x_pos, conv_df['slope'], color=colors, alpha=0.7)
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(conv_df['config'], rotation=45, ha='right')
                ax4.set_ylabel('Convergence Rate (KL slope)')
                ax4.set_title(f'Grid {grid}: Convergence Analysis')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax4.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, slope, n_trials in zip(bars, conv_df['slope'], conv_df['n_trials']):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., 
                            height + (0.001 if height >= 0 else -0.003),
                            f'{slope:.4f}\n(n={n_trials})', ha='center', 
                            va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.suptitle('KL Divergence Evolution by Grid Size', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        evolution_grid_path = self.output_dir / 'kl_evolution_by_grid_size.png'
        plt.savefig(evolution_grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"KL evolution by grid size saved: {evolution_grid_path}")
    
    def create_evolution_by_agent_count(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Show KL evolution for each agent count across grids and intervals"""
        
        n_agents = len(agent_numbers)
        fig = plt.figure(figsize=(24, 6 * n_agents))
        
        colors_interval = plt.cm.viridis(np.linspace(0, 1, len(merge_intervals)))
        colors_grid = plt.cm.plasma(np.linspace(0, 1, len(grid_sizes)))
        
        for i, agents in enumerate(agent_numbers):
            
            # Plot 1: All intervals for this agent count
            ax1 = plt.subplot(n_agents, 4, i*4 + 1)
            
            for j, interval in enumerate(merge_intervals):
                interval_series = []
                for grid in grid_sizes:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        interval_series.extend(series_list)
                
                if interval_series:
                    max_len = max(len(series) for series in interval_series)
                    padded_series = []
                    
                    for series in interval_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    int_str = f"Int {int(interval)}" if interval != float('inf') else "No Merge"
                    ax1.plot(steps, mean_series, color=colors_interval[j], linewidth=2, 
                            label=f'{int_str} (n={len(interval_series)})', alpha=0.8)
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('KL Divergence to Ground Truth')
            ax1.set_title(f'{agents} Agents: Evolution by Merge Interval')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: All grid sizes for this agent count
            ax2 = plt.subplot(n_agents, 4, i*4 + 2)
            
            for j, grid in enumerate(grid_sizes):
                grid_series = []
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        grid_series.extend(series_list)
                
                if grid_series:
                    max_len = max(len(series) for series in grid_series)
                    padded_series = []
                    
                    for series in grid_series:
                        if len(series) < max_len:
                            padded = list(series) + [series[-1]] * (max_len - len(series))
                        else:
                            padded = series[:max_len]
                        padded_series.append(padded)
                    
                    mean_series = np.mean(padded_series, axis=0)
                    steps = range(len(mean_series))
                    
                    ax2.plot(steps, mean_series, color=colors_grid[j], linewidth=2, 
                            label=f'Grid {grid} (n={len(grid_series)})', alpha=0.8)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('KL Divergence to Ground Truth')
            ax2.set_title(f'{agents} Agents: Evolution by Grid Size')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Phase analysis (early vs late performance)
            ax3 = plt.subplot(n_agents, 4, i*4 + 3)
            
            # Calculate early vs late performance for each configuration
            phase_data = []
            
            for grid in grid_sizes:
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        early_perfs = []
                        late_perfs = []
                        
                        for series in series_list:
                            if len(series) > 20:
                                early_quarter = len(series) // 4
                                late_quarter = 3 * len(series) // 4
                                
                                early_perf = np.mean(series[:early_quarter])
                                late_perf = np.mean(series[late_quarter:])
                                
                                early_perfs.append(early_perf)
                                late_perfs.append(late_perf)
                        
                        if early_perfs and late_perfs:
                            int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                            phase_data.append({
                                'config': f'{grid}-int{int_str}',
                                'early': np.mean(early_perfs),
                                'late': np.mean(late_perfs),
                                'improvement': np.mean(early_perfs) - np.mean(late_perfs)
                            })
            
            if phase_data:
                phase_df = pd.DataFrame(phase_data)
                
                # Scatter plot: early vs late performance
                colors = ['green' if imp > 0 else 'red' for imp in phase_df['improvement']]
                scatter = ax3.scatter(phase_df['early'], phase_df['late'], 
                                    c=colors, s=100, alpha=0.7, edgecolors='black')
                
                
                # Add diagonal line (no improvement)
                min_val = min(phase_df['early'].min(), phase_df['late'].min())
                max_val = max(phase_df['early'].max(), phase_df['late'].max())
                ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Change')
                
                # Add labels for each point
                for _, row in phase_df.iterrows():
                    ax3.annotate(row['config'], (row['early'], row['late']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
                
                ax3.set_xlabel('Early Performance (First Quarter)')
                ax3.set_ylabel('Late Performance (Last Quarter)')
                ax3.set_title(f'{agents} Agents: Early vs Late Performance')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Detailed convergence analysis
            ax4 = plt.subplot(n_agents, 4, i*4 + 4)
            
            # Show convergence patterns for each interval
            convergence_patterns = {}
            
            for interval in merge_intervals:
                interval_slopes = []
                for grid in grid_sizes:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for series in series_list:
                            if len(series) > 10:
                                # Calculate slope in multiple segments
                                n_segments = 3
                                segment_size = len(series) // n_segments
                                
                                for seg in range(n_segments):
                                    start_idx = seg * segment_size
                                    end_idx = min((seg + 1) * segment_size, len(series))
                                    
                                    if end_idx - start_idx > 2:
                                        y = series[start_idx:end_idx]
                                        x = range(len(y))
                                        slope, _ = np.polyfit(x, y, 1)
                                        interval_slopes.append(slope)
                
                if interval_slopes:
                    convergence_patterns[interval] = interval_slopes
            
            # Box plot of convergence rates
            if convergence_patterns:
                box_data = []
                box_labels = []
                
                for interval in sorted(convergence_patterns.keys()):
                    box_data.append(convergence_patterns[interval])
                    int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                    box_labels.append(f'Int {int_str}')
                
                bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                # Color boxes by median convergence rate
                for patch, slopes in zip(bp['boxes'], box_data):
                    median_slope = np.median(slopes)
                    color = 'green' if median_slope < 0 else 'red'
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax4.set_xlabel('Merge Interval')
                ax4.set_ylabel('Convergence Rate Distribution')
                ax4.set_title(f'{agents} Agents: Convergence Rate Patterns')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax4.grid(True, alpha=0.3)
        
        plt.suptitle('KL Divergence Evolution by Agent Count', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        evolution_agent_path = self.output_dir / 'kl_evolution_by_agent_count.png'
        plt.savefig(evolution_agent_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"KL evolution by agent count saved: {evolution_agent_path}")
    
    def create_evolution_overview(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Create comprehensive overview of evolution patterns"""
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Global evolution patterns
        ax1 = plt.subplot(3, 4, 1)
        
        # Average across ALL conditions for each interval
        colors_interval = plt.cm.viridis(np.linspace(0, 1, len(merge_intervals)))
        
        for j, interval in enumerate(merge_intervals):
            all_interval_series = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        all_interval_series.extend(series_list)
            
            if all_interval_series:
                max_len = max(len(series) for series in all_interval_series)
                padded_series = []
                
                for series in all_interval_series:
                    if len(series) < max_len:
                        padded = list(series) + [series[-1]] * (max_len - len(series))
                    else:
                        padded = series[:max_len]
                    padded_series.append(padded)
                
                mean_series = np.mean(padded_series, axis=0)
                steps = range(len(mean_series))
                
                int_str = f"Int {int(interval)}" if interval != float('inf') else "No Merge"
                ax1.plot(steps, mean_series, color=colors_interval[j], linewidth=3, 
                        label=f'{int_str} (n={len(all_interval_series)})', alpha=0.9)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('KL Divergence to Ground Truth')
        ax1.set_title('Global Evolution Patterns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Convergence rate heatmap (Grid × Agents)
        ax2 = plt.subplot(3, 4, 2)
        
        convergence_matrix = np.zeros((len(grid_sizes), len(agent_numbers)))
        
        for i, grid in enumerate(grid_sizes):
            for j, agents in enumerate(agent_numbers):
                all_slopes = []
                
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for series in series_list:
                            if len(series) > 10:
                                half_point = len(series) // 2
                                y = series[half_point:]
                                x = range(len(y))
                                if len(y) > 1:
                                    slope, _ = np.polyfit(x, y, 1)
                                    all_slopes.append(slope)
                
                if all_slopes:
                    convergence_matrix[i, j] = np.mean(all_slopes)
        
        im = ax2.imshow(convergence_matrix, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xticks(range(len(agent_numbers)))
        ax2.set_yticks(range(len(grid_sizes)))
        ax2.set_xticklabels([f'{agents}ag' for agents in agent_numbers])
        ax2.set_yticklabels([f'{grid}' for grid in grid_sizes])
        ax2.set_title('Average Convergence Rate\n(Green=Converging, Red=Diverging)')
        ax2.set_xlabel('Number of Agents')
        ax2.set_ylabel('Grid Size')
        
        # Add text annotations
        for i in range(len(grid_sizes)):
            for j in range(len(agent_numbers)):
                text = f'{convergence_matrix[i, j]:.3f}'
                color = 'white' if abs(convergence_matrix[i, j]) > 0.01 else 'black'
                ax2.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. Performance improvement over time
        ax3 = plt.subplot(3, 4, 3)
        
        # Calculate relative improvement from initial performance
        improvement_data = {}
        
        for interval in merge_intervals:
            all_improvements = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for series in series_list:
                            if len(series) > 10:
                                initial = np.mean(series[:5])  # First 5 steps
                                final = np.mean(series[-5:])   # Last 5 steps
                                
                                if initial > 0:
                                    improvement = (initial - final) / initial * 100
                                    all_improvements.append(improvement)
            
            if all_improvements:
                improvement_data[interval] = all_improvements
        
        # Box plot of improvements
        if improvement_data:
            box_data = []
            box_labels = []
            
            for interval in sorted(improvement_data.keys()):
                box_data.append(improvement_data[interval])
                int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                box_labels.append(f'Int {int_str}')
            
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color by median improvement
            for patch, improvements in zip(bp['boxes'], box_data):
                median_imp = np.median(improvements)
                color = 'green' if median_imp > 0 else 'red'
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_xlabel('Merge Interval')
            ax3.set_ylabel('Performance Improvement (%)')
            ax3.set_title('Performance Improvement Distribution\n(Initial vs Final)')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3)
        
        # 4. Evolution variance analysis
        ax4 = plt.subplot(3, 4, 4)
        
        # Calculate how much variance there is in evolution patterns
        variance_data = {}
        
        for interval in merge_intervals:
            all_variances = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        if len(series_list) > 1:
                            # Calculate variance at each time step
                            max_len = max(len(series) for series in series_list)
                            padded_series = []
                            
                            for series in series_list:
                                if len(series) < max_len:
                                    padded = list(series) + [series[-1]] * (max_len - len(series))
                                else:
                                    padded = series[:max_len]
                                padded_series.append(padded)
                            
                            # Average variance across time
                            time_variances = np.var(padded_series, axis=0)
                            avg_variance = np.mean(time_variances)
                            all_variances.append(avg_variance)
            
            if all_variances:
                variance_data[interval] = np.mean(all_variances)
        
        if variance_data:
            intervals = list(variance_data.keys())
            variances = list(variance_data.values())
            
            colors = [colors_interval[list(merge_intervals).index(interval)] for interval in intervals]
            bars = ax4.bar(range(len(intervals)), variances, color=colors, alpha=0.7)
            
            ax4.set_xticks(range(len(intervals)))
            ax4.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in intervals])
            ax4.set_xlabel('Merge Interval')
            ax4.set_ylabel('Average Evolution Variance')
            ax4.set_title('Consistency of Evolution Patterns\n(Lower = More Consistent)')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, var in zip(bars, variances):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{var:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 5-8. Detailed phase analysis
        phases = ['Early (0-25%)', 'Mid-Early (25-50%)', 'Mid-Late (50-75%)', 'Late (75-100%)']
        
        for phase_idx, phase_name in enumerate(phases):
            ax = plt.subplot(3, 4, 5 + phase_idx)
            
            phase_performance = {}
            
            for interval in merge_intervals:
                phase_values = []
                
                for grid in grid_sizes:
                    for agents in agent_numbers:
                        if (grid in time_series_data and 
                            agents in time_series_data[grid] and 
                            interval in time_series_data[grid][agents]):
                            series_list = time_series_data[grid][agents][interval]['kl_series']
                            
                            for series in series_list:
                                if len(series) > 20:
                                    quarter_size = len(series) // 4
                                    start_idx = phase_idx * quarter_size
                                    end_idx = min((phase_idx + 1) * quarter_size, len(series))
                                    
                                    if end_idx > start_idx:
                                        phase_avg = np.mean(series[start_idx:end_idx])
                                        phase_values.append(phase_avg)
                
                if phase_values:
                    phase_performance[interval] = np.mean(phase_values)
            
            if phase_performance:
                intervals = list(phase_performance.keys())
                performances = list(phase_performance.values())
                
                colors = [colors_interval[list(merge_intervals).index(interval)] for interval in intervals]
                bars = ax.bar(range(len(intervals)), performances, color=colors, alpha=0.7)
                
                ax.set_xticks(range(len(intervals)))
                ax.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in intervals])
                ax.set_xlabel('Merge Interval')
                ax.set_ylabel('Average KL Divergence')
                ax.set_title(f'{phase_name} Performance')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, perf in zip(bars, performances):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{perf:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 9. Summary statistics table
        ax9 = plt.subplot(3, 4, 9)
        ax9.axis('off')
        
        # Calculate key statistics
        summary_stats = []
        
        for interval in merge_intervals:
            all_series = []
            all_slopes = []
            all_improvements = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for series in series_list:
                            all_series.append(series)
                            
                            if len(series) > 10:
                                # Slope
                                half_point = len(series) // 2
                                y = series[half_point:]
                                x = range(len(y))
                                if len(y) > 1:
                                    slope, _ = np.polyfit(x, y, 1)
                                    all_slopes.append(slope)
                                
                                # Improvement
                                initial = np.mean(series[:5])
                                final = np.mean(series[-5:])
                                if initial > 0:
                                    improvement = (initial - final) / initial * 100
                                    all_improvements.append(improvement)
            
            if all_series:
                avg_performance = np.mean([np.mean(series) for series in all_series])
                avg_slope = np.mean(all_slopes) if all_slopes else 0
                avg_improvement = np.mean(all_improvements) if all_improvements else 0
                
                int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                summary_stats.append([
                    int_str,
                    f"{avg_performance:.4f}",
                    f"{avg_slope:+.4f}",
                    f"{avg_improvement:+.1f}%",
                    f"{len(all_series)}"
                ])
        
        if summary_stats:
            table = ax9.table(
                cellText=summary_stats,
                colLabels=['Interval', 'Avg KL', 'Slope', 'Improvement', 'N Trials'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0.2, 1, 0.6]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Highlight best performing interval
            best_row_idx = min(range(len(summary_stats)), 
                             key=lambda i: float(summary_stats[i][1])) + 1
            
            for j in range(len(summary_stats[0])):
                table[(best_row_idx, j)].set_facecolor('lightgreen')
                table[(best_row_idx, j)].set_alpha(0.5)
        
        ax9.set_title('Evolution Summary Statistics', fontsize=12, fontweight='bold', y=0.9)
        
        # 10-12. Best configuration analysis
        best_configs = []
        
        # Find best configuration for each grid size
        for grid in grid_sizes:
            grid_configs = []
            
            for agents in agent_numbers:
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        if series_list:
                            avg_performance = np.mean([np.mean(series) for series in series_list])
                            grid_configs.append({
                                'grid': grid,
                                'agents': agents,
                                'interval': interval,
                                'performance': avg_performance,
                                'series_list': series_list
                            })
            
            if grid_configs:
                best_config = min(grid_configs, key=lambda x: x['performance'])
                best_configs.append(best_config)
        
        # Plot best configuration evolution for each grid
        for plot_idx, config in enumerate(best_configs):
            if plot_idx < 3:  # Only plot first 3 grids
                ax = plt.subplot(3, 4, 10 + plot_idx)
                
                series_list = config['series_list']
                max_len = max(len(series) for series in series_list)
                padded_series = []
                
                for series in series_list:
                    if len(series) < max_len:
                        padded = list(series) + [series[-1]] * (max_len - len(series))
                    else:
                        padded = series[:max_len]
                    padded_series.append(padded)
                
                mean_series = np.mean(padded_series, axis=0)
                std_series = np.std(padded_series, axis=0)
                steps = range(len(mean_series))
                
                ax.plot(steps, mean_series, 'b-', linewidth=3, label='Mean')
                ax.fill_between(steps, mean_series - std_series, mean_series + std_series,
                               alpha=0.3, color='blue')
                
                # Show individual trials as thin lines
                for series in padded_series[:5]:  # Show max 5 individual trials
                    ax.plot(steps, series, alpha=0.2, linewidth=1, color='gray')
                
                int_str = f"{int(config['interval'])}" if config['interval'] != float('inf') else "∞"
                ax.set_title(f"Best for Grid {config['grid']}:\n{config['agents']}ag, int{int_str}")
                ax.set_xlabel('Time Step')
                ax.set_ylabel('KL Divergence')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive KL Divergence Evolution Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        evolution_overview_path = self.output_dir / 'kl_evolution_overview.png'
        plt.savefig(evolution_overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"KL evolution overview saved: {evolution_overview_path}")

    def create_information_tracking_analysis(self):
        """Analyze cumulative information gain vs tracking error - AVERAGED ACROSS CONDITIONS"""
        
        print("\nCreating information tracking analysis...")
        
        # Extract merge events with information metrics
        merge_data = []
        
        for _, row in self.df.iterrows():
            trial = row['trial_data']
            
            if 'time_series' not in trial:
                continue
                
            ts = trial['time_series']
            
            if ('cumulative_info_since_merge' not in ts or 
                'ground_truth_evolution' not in ts or
                'kl_divergence_to_truth' not in ts):
                continue
            
            # Extract data at merge events
            if 'merge_events' in ts and ts['merge_events']:
                for merge_event in ts['merge_events']:
                    merge_step = merge_event['step']
                    
                    if merge_step < len(ts['cumulative_info_since_merge']):
                        merge_data.append({
                            'grid_size': row['grid_size'],
                            'n_agents': row['n_agents'],
                            'merge_interval': row['merge_interval'],
                            'merge_method': row.get('merge_method', 'standard_kl'),
                            'step': merge_step,
                            'cumulative_info': ts['cumulative_info_since_merge'][merge_step],
                            'tracking_error': ts['kl_divergence_to_truth'][merge_step],
                            'kl_before': merge_event['kl_before'],
                            'kl_after': merge_event['kl_after'],
                            'improvement': merge_event['improvement'],
                            'steps_since_merge': ts['steps_since_merge'][merge_step]
                        })
        
        if not merge_data:
            print("  WARNING: No information tracking data found!")
            print("  Make sure you've run experiments with the updated code.")
            return
        
        merge_df = pd.DataFrame(merge_data)
        print(f"  Found {len(merge_df)} merge events with information tracking")
        
        # Calculate efficiency metric
        merge_df['efficiency'] = merge_df['improvement'] / (merge_df['cumulative_info'] + 1e-6)
        merge_df['info_rate'] = merge_df['cumulative_info'] / (merge_df['steps_since_merge'] + 1)
        
        # Clean up any inf/nan values that could cause rendering issues
        merge_df['efficiency'] = merge_df['efficiency'].replace([np.inf, -np.inf], np.nan)
        merge_df['info_rate'] = merge_df['info_rate'].replace([np.inf, -np.inf], np.nan)
        
        # Create comprehensive averaged visualization
        fig = plt.figure(figsize=(20, 12), dpi=100)
        
        # ========== ROW 1: BY MERGE INTERVAL ==========
        
        # 1. Cumulative Info at Merge by Interval
        ax1 = plt.subplot(3, 4, 1)
        
        interval_stats = merge_df.groupby('merge_interval').agg({
            'cumulative_info': ['mean', 'std'],
            'tracking_error': ['mean', 'std'],
            'improvement': ['mean', 'std']
        })
        
        intervals = sorted(merge_df['merge_interval'].unique())
        x_labels = [f"{int(i)}" if i != float('inf') else "∞" for i in intervals]
        x_pos = range(len(intervals))
        
        means = [interval_stats.loc[i, ('cumulative_info', 'mean')] for i in intervals]
        stds = [interval_stats.loc[i, ('cumulative_info', 'std')] for i in intervals]
        
        ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        ax1.set_xlabel('Merge Interval')
        ax1.set_ylabel('Cumulative Information at Merge')
        ax1.set_title('Info Accumulated Before Communication')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (m, s) in enumerate(zip(means, stds)):
            ax1.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        # 2. Tracking Error at Merge by Interval
        ax2 = plt.subplot(3, 4, 2)
        
        means = [interval_stats.loc[i, ('tracking_error', 'mean')] for i in intervals]
        stds = [interval_stats.loc[i, ('tracking_error', 'std')] for i in intervals]
        
        ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.set_xlabel('Merge Interval')
        ax2.set_ylabel('Tracking Error at Merge')
        ax2.set_title('How Wrong Before Communication')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (m, s) in enumerate(zip(means, stds)):
            ax2.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        # 3. Improvement from Merge by Interval
        ax3 = plt.subplot(3, 4, 3)
        
        means = [interval_stats.loc[i, ('improvement', 'mean')] for i in intervals]
        stds = [interval_stats.loc[i, ('improvement', 'std')] for i in intervals]
        
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_labels)
        ax3.set_xlabel('Merge Interval')
        ax3.set_ylabel('KL Improvement from Merge')
        ax3.set_title('Benefit of Communication')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, (m, s) in enumerate(zip(means, stds)):
            ax3.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        # 4. Efficiency by Interval
        ax4 = plt.subplot(3, 4, 4)
        
        efficiency_stats = merge_df.groupby('merge_interval')['efficiency'].agg(['mean', 'std'])
        means = [efficiency_stats.loc[i, 'mean'] for i in intervals]
        stds = [efficiency_stats.loc[i, 'std'] for i in intervals]
        
        colors = ['green' if m > 0 else 'red' for m in means]
        ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(x_labels)
        ax4.set_xlabel('Merge Interval')
        ax4.set_ylabel('Efficiency (Improvement / Info)')
        ax4.set_title('Communication Efficiency')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, (m, s) in enumerate(zip(means, stds)):
            y_offset = s + 0.01 if m >= 0 else -(s + 0.01)
            ax4.text(i, m + y_offset, f'{m:.2f}', ha='center', fontsize=9, fontweight='bold')
        
        # ========== ROW 2: BY GRID SIZE ==========
        
        # 5. Cumulative Info by Grid Size
        ax5 = plt.subplot(3, 4, 5)
        
        grid_stats = merge_df.groupby('grid_size').agg({
            'cumulative_info': ['mean', 'std'],
            'tracking_error': ['mean', 'std'],
            'improvement': ['mean', 'std']
        })
        
        grids = sorted(merge_df['grid_size'].unique())
        x_pos = range(len(grids))
        
        means = [grid_stats.loc[g, ('cumulative_info', 'mean')] for g in grids]
        stds = [grid_stats.loc[g, ('cumulative_info', 'std')] for g in grids]
        
        ax5.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='red', edgecolor='black')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(grids, rotation=45)
        ax5.set_xlabel('Grid Size')
        ax5.set_ylabel('Cumulative Information at Merge')
        ax5.set_title('Info by Problem Size')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Tracking Error by Grid Size
        ax6 = plt.subplot(3, 4, 6)
        
        means = [grid_stats.loc[g, ('tracking_error', 'mean')] for g in grids]
        stds = [grid_stats.loc[g, ('tracking_error', 'std')] for g in grids]
        
        ax6.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='blue', edgecolor='black')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(grids, rotation=45)
        ax6.set_xlabel('Grid Size')
        ax6.set_ylabel('Tracking Error at Merge')
        ax6.set_title('Error by Problem Size')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Improvement by Grid Size
        ax7 = plt.subplot(3, 4, 7)
        
        means = [grid_stats.loc[g, ('improvement', 'mean')] for g in grids]
        stds = [grid_stats.loc[g, ('improvement', 'std')] for g in grids]
        
        ax7.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='green', edgecolor='black')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(grids, rotation=45)
        ax7.set_xlabel('Grid Size')
        ax7.set_ylabel('Improvement from Merge')
        ax7.set_title('Merge Benefit by Problem Size')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Info Accumulation Rate by Grid Size
        ax8 = plt.subplot(3, 4, 8)
        
        rate_stats = merge_df.groupby('grid_size')['info_rate'].agg(['mean', 'std'])
        means = [rate_stats.loc[g, 'mean'] for g in grids]
        stds = [rate_stats.loc[g, 'std'] for g in grids]
        
        ax8.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange', edgecolor='black')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(grids, rotation=45)
        ax8.set_xlabel('Grid Size')
        ax8.set_ylabel('Info Accumulation Rate (per step)')
        ax8.set_title('How Fast Info Accumulates')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # ========== ROW 3: BY AGENT COUNT ==========
        
        # 9. Cumulative Info by Agent Count
        ax9 = plt.subplot(3, 4, 9)
        
        agent_stats = merge_df.groupby('n_agents').agg({
            'cumulative_info': ['mean', 'std'],
            'tracking_error': ['mean', 'std'],
            'improvement': ['mean', 'std']
        })
        
        agents = sorted(merge_df['n_agents'].unique())
        x_pos = range(len(agents))
        
        means = [agent_stats.loc[a, ('cumulative_info', 'mean')] for a in agents]
        stds = [agent_stats.loc[a, ('cumulative_info', 'std')] for a in agents]
        
        ax9.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='red', edgecolor='black')
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels([f'{a} agents' for a in agents])
        ax9.set_xlabel('Number of Agents')
        ax9.set_ylabel('Cumulative Information at Merge')
        ax9.set_title('Info by Team Size')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Tracking Error by Agent Count
        ax10 = plt.subplot(3, 4, 10)
        
        means = [agent_stats.loc[a, ('tracking_error', 'mean')] for a in agents]
        stds = [agent_stats.loc[a, ('tracking_error', 'std')] for a in agents]
        
        ax10.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='blue', edgecolor='black')
        ax10.set_xticks(x_pos)
        ax10.set_xticklabels([f'{a} agents' for a in agents])
        ax10.set_xlabel('Number of Agents')
        ax10.set_ylabel('Tracking Error at Merge')
        ax10.set_title('Error by Team Size')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 11. Improvement by Agent Count
        ax11 = plt.subplot(3, 4, 11)
        
        means = [agent_stats.loc[a, ('improvement', 'mean')] for a in agents]
        stds = [agent_stats.loc[a, ('improvement', 'std')] for a in agents]
        
        ax11.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='green', edgecolor='black')
        ax11.set_xticks(x_pos)
        ax11.set_xticklabels([f'{a} agents' for a in agents])
        ax11.set_xlabel('Number of Agents')
        ax11.set_ylabel('Improvement from Merge')
        ax11.set_title('Merge Benefit by Team Size')
        ax11.grid(True, alpha=0.3, axis='y')
        
        # 12. Combined Heatmap: Interval × Grid Size
        ax12 = plt.subplot(3, 4, 12)
        
        try:
            # Create pivot table for heatmap
            heatmap_data = merge_df.groupby(['merge_interval', 'grid_size'])['efficiency'].mean().reset_index()
            
            if not heatmap_data.empty:
                pivot_table = heatmap_data.pivot(index='merge_interval', columns='grid_size', values='efficiency')
                
                # Clean any problematic values
                pivot_table = pivot_table.fillna(0)
                
                # Rename index for display
                pivot_table.index = [f"{int(i)}" if i != float('inf') else "∞" for i in pivot_table.index]
                
                sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', 
                           center=0, ax=ax12, cbar_kws={'label': 'Efficiency'})
                ax12.set_title('Efficiency: Interval × Grid Size')
                ax12.set_xlabel('Grid Size')
                ax12.set_ylabel('Merge Interval')
        except Exception as e:
            ax12.text(0.5, 0.5, f'Heatmap Error:\n{str(e)}', 
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Efficiency Heatmap (Error)')
        
        plt.suptitle('Information Tracking Analysis: When Should Agents Communicate?', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save with safer parameters
        info_tracking_path = self.output_dir / 'information_tracking_analysis.png'
        plt.savefig(info_tracking_path, dpi=150, bbox_inches=None)
        plt.close()
        
        print(f"  Information tracking analysis saved: {info_tracking_path}")
        
        # Print detailed statistics
        print(f"\n  ========== KEY INSIGHTS ==========")
        
        print(f"\n  BY MERGE INTERVAL:")
        for interval in intervals:
            stats = interval_stats.loc[interval]
            int_str = f"{int(interval)}" if interval != float('inf') else "∞"
            print(f"    Interval {int_str}:")
            print(f"      Cumulative Info: {stats[('cumulative_info', 'mean')]:.4f} ± {stats[('cumulative_info', 'std')]:.4f}")
            print(f"      Tracking Error:  {stats[('tracking_error', 'mean')]:.4f} ± {stats[('tracking_error', 'std')]:.4f}")
            print(f"      Improvement:     {stats[('improvement', 'mean')]:.4f} ± {stats[('improvement', 'std')]:.4f}")
        
        print(f"\n  BY GRID SIZE:")
        for grid in grids:
            stats = grid_stats.loc[grid]
            print(f"    Grid {grid}:")
            print(f"      Cumulative Info: {stats[('cumulative_info', 'mean')]:.4f} ± {stats[('cumulative_info', 'std')]:.4f}")
            print(f"      Tracking Error:  {stats[('tracking_error', 'mean')]:.4f} ± {stats[('tracking_error', 'std')]:.4f}")
        
        print(f"\n  BY AGENT COUNT:")
        for agents in sorted(merge_df['n_agents'].unique()):
            stats = agent_stats.loc[agents]
            print(f"    {agents} agents:")
            print(f"      Cumulative Info: {stats[('cumulative_info', 'mean')]:.4f} ± {stats[('cumulative_info', 'std')]:.4f}")
            print(f"      Tracking Error:  {stats[('tracking_error', 'mean')]:.4f} ± {stats[('tracking_error', 'std')]:.4f}")
        
        # Find best configurations
        best_efficiency_idx = merge_df.groupby(['merge_interval', 'grid_size', 'n_agents'])['efficiency'].mean().idxmax()
        print(f"\n  BEST CONFIGURATION (highest efficiency):")
        print(f"    Interval: {int(best_efficiency_idx[0]) if best_efficiency_idx[0] != float('inf') else '∞'}")
        print(f"    Grid: {best_efficiency_idx[1]}")
        print(f"    Agents: {best_efficiency_idx[2]}")
        
        print(f"\n  ===================================")

    def create_cumulative_information_gain_analysis(self):
        """
        Analyze cumulative information gain over time
        - With reset at merge events (local accumulation)
        - Without reset (global accumulation)
        """
        print("Creating cumulative information gain analysis...")
        
        time_series_data = self.extract_time_series_data()
        
        if not time_series_data:
            print("No time series data available for cumulative analysis")
            return
        
        grid_sizes = sorted(time_series_data.keys())
        agent_numbers = sorted(set(agents for grid_data in time_series_data.values() 
                                 for agents in grid_data.keys()))
        merge_intervals = sorted(set(interval for grid_data in time_series_data.values() 
                                   for agent_data in grid_data.values() 
                                   for interval in agent_data.keys()))
        
        # Create multiple comprehensive figures
        self.create_cumulative_by_interval(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        self.create_cumulative_by_grid(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        self.create_cumulative_by_agents(time_series_data, grid_sizes, agent_numbers, merge_intervals)
        self.create_cumulative_comparison_matrix(time_series_data, grid_sizes, agent_numbers, merge_intervals)
    
    def create_cumulative_by_interval(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Cumulative information gain analysis by merge interval"""
        
        n_intervals = len(merge_intervals)
        fig = plt.figure(figsize=(24, 8 * n_intervals))
        
        for i, interval in enumerate(merge_intervals):
            
            # Plot 1: Cumulative with resets (local)
            ax1 = plt.subplot(n_intervals, 3, i*3 + 1)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(grid_sizes) * len(agent_numbers)))
            color_idx = 0
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        if series_list:
                            # Compute cumulative info gain with resets
                            all_cumulative_reset = []
                            
                            for kl_series in series_list:
                                cumulative_reset = [0]
                                current_cum = 0
                                
                                # Reset at each merge interval
                                for step in range(1, len(kl_series)):
                                    # Information gain is reduction in KL
                                    info_gain = max(0, kl_series[step-1] - kl_series[step])
                                    current_cum += info_gain
                                    
                                    # Reset at merge events
                                    if interval != float('inf') and interval > 0 and step % interval == 0:
                                        current_cum = 0
                                    
                                    cumulative_reset.append(current_cum)
                                
                                all_cumulative_reset.append(cumulative_reset)
                            
                            # Average across trials
                            max_len = max(len(c) for c in all_cumulative_reset)
                            padded = []
                            for c in all_cumulative_reset:
                                if len(c) < max_len:
                                    padded.append(c + [c[-1]] * (max_len - len(c)))
                                else:
                                    padded.append(c)
                            
                            mean_cumulative = np.mean(padded, axis=0)
                            steps = range(len(mean_cumulative))
                            
                            label = f"Grid {grid}, {agents}ag"
                            ax1.plot(steps, mean_cumulative, color=colors[color_idx], 
                                   linewidth=2, alpha=0.8, label=label)
                            
                            # Mark merge events
                            if interval != float('inf') and interval > 0:
                                merge_points = range(int(interval), len(steps), int(interval))
                                for mp in merge_points[:5]:
                                    ax1.axvline(mp, color=colors[color_idx], alpha=0.2, 
                                              linestyle='--', linewidth=1)
                            
                            color_idx += 1
            
            int_str = f"Interval {int(interval)}" if interval != float('inf') else "No Merge"
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Cumulative Info Gain (Reset at Merge)')
            ax1.set_title(f'{int_str}: Local Accumulation')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cumulative without resets (global)
            ax2 = plt.subplot(n_intervals, 3, i*3 + 2)
            
            color_idx = 0
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        if series_list:
                            # Compute cumulative info gain without resets
                            all_cumulative_global = []
                            
                            for kl_series in series_list:
                                cumulative_global = [0]
                                total = 0
                                
                                for step in range(1, len(kl_series)):
                                    info_gain = max(0, kl_series[step-1] - kl_series[step])
                                    total += info_gain
                                    cumulative_global.append(total)
                                
                                all_cumulative_global.append(cumulative_global)
                            
                            # Average across trials
                            max_len = max(len(c) for c in all_cumulative_global)
                            padded = []
                            for c in all_cumulative_global:
                                if len(c) < max_len:
                                    padded.append(c + [c[-1]] * (max_len - len(c)))
                                else:
                                    padded.append(c)
                            
                            mean_cumulative = np.mean(padded, axis=0)
                            steps = range(len(mean_cumulative))
                            
                            label = f"Grid {grid}, {agents}ag"
                            ax2.plot(steps, mean_cumulative, color=colors[color_idx], 
                                   linewidth=2, alpha=0.8, label=label)
                            
                            # Mark merge events (visual only, no reset)
                            if interval != float('inf') and interval > 0:
                                merge_points = range(int(interval), len(steps), int(interval))
                                for mp in merge_points[:5]:
                                    ax2.axvline(mp, color=colors[color_idx], alpha=0.1, 
                                              linestyle=':', linewidth=1)
                            
                            color_idx += 1
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Cumulative Info Gain (Never Reset)')
            ax2.set_title(f'{int_str}: Global Accumulation')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Rate of information gain
            ax3 = plt.subplot(n_intervals, 3, i*3 + 3)
            
            color_idx = 0
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        if series_list:
                            # Compute instantaneous info gain rate
                            all_rates = []
                            
                            for kl_series in series_list:
                                rates = [0]
                                for step in range(1, len(kl_series)):
                                    rate = max(0, kl_series[step-1] - kl_series[step])
                                    rates.append(rate)
                                all_rates.append(rates)
                            
                            # Average across trials
                            max_len = max(len(r) for r in all_rates)
                            padded = []
                            for r in all_rates:
                                if len(r) < max_len:
                                    padded.append(r + [r[-1]] * (max_len - len(r)))
                                else:
                                    padded.append(r)
                            
                            mean_rate = np.mean(padded, axis=0)
                            
                            # Smooth the rate for better visualization
                            window = 10
                            if len(mean_rate) >= window:
                                smoothed = np.convolve(mean_rate, np.ones(window)/window, mode='valid')
                                steps = range(window-1, len(mean_rate))
                            else:
                                smoothed = mean_rate
                                steps = range(len(mean_rate))
                            
                            label = f"Grid {grid}, {agents}ag"
                            ax3.plot(steps, smoothed, color=colors[color_idx], 
                                   linewidth=2, alpha=0.8, label=label)
                            
                            color_idx += 1
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Info Gain Rate (smoothed)')
            ax3.set_title(f'{int_str}: Information Gain Rate')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Cumulative Information Gain by Merge Interval', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'cumulative_info_gain_by_interval.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative info gain by interval saved: {output_path}")
    
    def create_cumulative_by_grid(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Cumulative information gain analysis by grid size"""
        
        n_grids = len(grid_sizes)
        fig = plt.figure(figsize=(24, 8 * n_grids))
        
        for i, grid in enumerate(grid_sizes):
            
            # Plot 1: Cumulative with resets for all intervals
            ax1 = plt.subplot(n_grids, 3, i*3 + 1)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(merge_intervals)))
            
            for j, interval in enumerate(merge_intervals):
                all_cumulative_reset = []
                
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            cumulative_reset = [0]
                            current_cum = 0
                            
                            for step in range(1, len(kl_series)):
                                info_gain = max(0, kl_series[step-1] - kl_series[step])
                                current_cum += info_gain
                                
                                if interval != float('inf') and interval > 0 and step % interval == 0:
                                    current_cum = 0
                                
                                cumulative_reset.append(current_cum)
                            
                            all_cumulative_reset.append(cumulative_reset)
                
                if all_cumulative_reset:
                    max_len = max(len(c) for c in all_cumulative_reset)
                    padded = []
                    for c in all_cumulative_reset:
                        if len(c) < max_len:
                            padded.append(c + [c[-1]] * (max_len - len(c)))
                        else:
                            padded.append(c)
                    
                    mean_cumulative = np.mean(padded, axis=0)
                    steps = range(len(mean_cumulative))
                    
                    int_str = f"Int {int(interval)}" if interval != float('inf') else "No Merge"
                    ax1.plot(steps, mean_cumulative, color=colors[j], 
                           linewidth=2, alpha=0.8, label=int_str)
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Cumulative Info Gain (Reset at Merge)')
            ax1.set_title(f'Grid {grid}: Local Accumulation by Interval')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cumulative without resets for all intervals
            ax2 = plt.subplot(n_grids, 3, i*3 + 2)
            
            for j, interval in enumerate(merge_intervals):
                all_cumulative_global = []
                
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            cumulative_global = [0]
                            total = 0
                            
                            for step in range(1, len(kl_series)):
                                info_gain = max(0, kl_series[step-1] - kl_series[step])
                                total += info_gain
                                cumulative_global.append(total)
                            
                            all_cumulative_global.append(cumulative_global)
                
                if all_cumulative_global:
                    max_len = max(len(c) for c in all_cumulative_global)
                    padded = []
                    for c in all_cumulative_global:
                        if len(c) < max_len:
                            padded.append(c + [c[-1]] * (max_len - len(c)))
                        else:
                            padded.append(c)
                    
                    mean_cumulative = np.mean(padded, axis=0)
                    steps = range(len(mean_cumulative))
                    
                    int_str = f"Int {int(interval)}" if interval != float('inf') else "No Merge"
                    ax2.plot(steps, mean_cumulative, color=colors[j], 
                           linewidth=2, alpha=0.8, label=int_str)
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Cumulative Info Gain (Never Reset)')
            ax2.set_title(f'Grid {grid}: Global Accumulation by Interval')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Final cumulative gain comparison
            ax3 = plt.subplot(n_grids, 3, i*3 + 3)
            
            final_gains_reset = []
            final_gains_global = []
            interval_labels = []
            
            for interval in merge_intervals:
                local_gains = []
                global_gains = []
                
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            # Local (with resets)
                            current_cum = 0
                            for step in range(1, len(kl_series)):
                                info_gain = max(0, kl_series[step-1] - kl_series[step])
                                current_cum += info_gain
                                if interval != float('inf') and interval > 0 and step % interval == 0:
                                    current_cum = 0
                            local_gains.append(current_cum)
                            
                            # Global (never reset)
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            global_gains.append(total)
                
                if local_gains and global_gains:
                    final_gains_reset.append(np.mean(local_gains))
                    final_gains_global.append(np.mean(global_gains))
                    int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                    interval_labels.append(int_str)
            
            x = np.arange(len(interval_labels))
            width = 0.35
            
            ax3.bar(x - width/2, final_gains_reset, width, label='Local (Reset)', alpha=0.8)
            ax3.bar(x + width/2, final_gains_global, width, label='Global (Never Reset)', alpha=0.8)
            
            ax3.set_xlabel('Merge Interval')
            ax3.set_ylabel('Final Cumulative Info Gain')
            ax3.set_title(f'Grid {grid}: Total Information Gained')
            ax3.set_xticks(x)
            ax3.set_xticklabels(interval_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Cumulative Information Gain by Grid Size', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'cumulative_info_gain_by_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative info gain by grid saved: {output_path}")
    
    def create_cumulative_by_agents(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Cumulative information gain analysis by agent count"""
        
        n_agents = len(agent_numbers)
        fig = plt.figure(figsize=(24, 8 * n_agents))
        
        for i, agents in enumerate(agent_numbers):
            
            # Plot 1: Cumulative with resets for all grids
            ax1 = plt.subplot(n_agents, 3, i*3 + 1)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(grid_sizes)))
            
            for j, grid in enumerate(grid_sizes):
                all_cumulative_reset = []
                
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            cumulative_reset = [0]
                            current_cum = 0
                            
                            for step in range(1, len(kl_series)):
                                info_gain = max(0, kl_series[step-1] - kl_series[step])
                                current_cum += info_gain
                                
                                if interval != float('inf') and interval > 0 and step % interval == 0:
                                    current_cum = 0
                                
                                cumulative_reset.append(current_cum)
                            
                            all_cumulative_reset.append(cumulative_reset)
                
                if all_cumulative_reset:
                    max_len = max(len(c) for c in all_cumulative_reset)
                    padded = []
                    for c in all_cumulative_reset:
                        if len(c) < max_len:
                            padded.append(c + [c[-1]] * (max_len - len(c)))
                        else:
                            padded.append(c)
                    
                    mean_cumulative = np.mean(padded, axis=0)
                    steps = range(len(mean_cumulative))
                    
                    ax1.plot(steps, mean_cumulative, color=colors[j], 
                           linewidth=2, alpha=0.8, label=f'Grid {grid}')
            
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Cumulative Info Gain (Reset at Merge)')
            ax1.set_title(f'{agents} Agents: Local Accumulation by Grid')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cumulative without resets for all grids
            ax2 = plt.subplot(n_agents, 3, i*3 + 2)
            
            for j, grid in enumerate(grid_sizes):
                all_cumulative_global = []
                
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            cumulative_global = [0]
                            total = 0
                            
                            for step in range(1, len(kl_series)):
                                info_gain = max(0, kl_series[step-1] - kl_series[step])
                                total += info_gain
                                cumulative_global.append(total)
                            
                            all_cumulative_global.append(cumulative_global)
                
                if all_cumulative_global:
                    max_len = max(len(c) for c in all_cumulative_global)
                    padded = []
                    for c in all_cumulative_global:
                        if len(c) < max_len:
                            padded.append(c + [c[-1]] * (max_len - len(c)))
                        else:
                            padded.append(c)
                    
                    mean_cumulative = np.mean(padded, axis=0)
                    steps = range(len(mean_cumulative))
                    
                    ax2.plot(steps, mean_cumulative, color=colors[j], 
                           linewidth=2, alpha=0.8, label=f'Grid {grid}')
            
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Cumulative Info Gain (Never Reset)')
            ax2.set_title(f'{agents} Agents: Global Accumulation by Grid')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Efficiency per agent
            ax3 = plt.subplot(n_agents, 3, i*3 + 3)
            
            for j, grid in enumerate(grid_sizes):
                interval_efficiencies = []
                interval_labels = []
                
                for interval in merge_intervals:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        total_gains = []
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            total_gains.append(total)
                        
                        if total_gains:
                            # Info gain per agent
                            efficiency = np.mean(total_gains) / agents
                            interval_efficiencies.append(efficiency)
                            int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                            interval_labels.append(int_str)
                
                if interval_efficiencies:
                    ax3.plot(range(len(interval_labels)), interval_efficiencies, 
                           'o-', color=colors[j], linewidth=2, markersize=8, 
                           alpha=0.8, label=f'Grid {grid}')
            
            ax3.set_xlabel('Merge Interval')
            ax3.set_ylabel('Info Gain per Agent')
            ax3.set_title(f'{agents} Agents: Information Efficiency')
            ax3.set_xticks(range(len(interval_labels)))
            ax3.set_xticklabels(interval_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Cumulative Information Gain by Agent Count', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'cumulative_info_gain_by_agents.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative info gain by agents saved: {output_path}")
    
    def create_cumulative_comparison_matrix(self, time_series_data, grid_sizes, agent_numbers, merge_intervals):
        """Create comprehensive comparison matrix of cumulative gains"""
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Heatmap: Grid × Interval (avg across agents) - Global accumulation
        ax1 = plt.subplot(2, 3, 1)
        
        matrix_data = np.zeros((len(grid_sizes), len(merge_intervals)))
        
        for i, grid in enumerate(grid_sizes):
            for j, interval in enumerate(merge_intervals):
                gains = []
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            gains.append(total)
                
                if gains:
                    matrix_data[i, j] = np.mean(gains)
        
        im = ax1.imshow(matrix_data, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(merge_intervals)))
        ax1.set_yticks(range(len(grid_sizes)))
        ax1.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        ax1.set_yticklabels([f"{g}" for g in grid_sizes])
        ax1.set_title('Total Info Gain: Grid × Interval\n(Avg across agents)')
        ax1.set_xlabel('Merge Interval')
        ax1.set_ylabel('Grid Size')
        
        for i in range(len(grid_sizes)):
            for j in range(len(merge_intervals)):
                text = ax1.text(j, i, f'{matrix_data[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Heatmap: Agents × Interval (avg across grids) - Global accumulation
        ax2 = plt.subplot(2, 3, 2)
        
        matrix_data2 = np.zeros((len(agent_numbers), len(merge_intervals)))
        
        for i, agents in enumerate(agent_numbers):
            for j, interval in enumerate(merge_intervals):
                gains = []
                for grid in grid_sizes:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            gains.append(total)
                
                if gains:
                    matrix_data2[i, j] = np.mean(gains)
        
        im = ax2.imshow(matrix_data2, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(merge_intervals)))
        ax2.set_yticks(range(len(agent_numbers)))
        ax2.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        ax2.set_yticklabels([f"{a}" for a in agent_numbers])
        ax2.set_title('Total Info Gain: Agents × Interval\n(Avg across grids)')
        ax2.set_xlabel('Merge Interval')
        ax2.set_ylabel('Number of Agents')
        
        for i in range(len(agent_numbers)):
            for j in range(len(merge_intervals)):
                text = ax2.text(j, i, f'{matrix_data2[i, j]:.2f}',
                              ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. Bar chart: Best configuration for each interval
        ax3 = plt.subplot(2, 3, 3)
        
        best_configs = []
        for interval in merge_intervals:
            max_gain = 0
            best_config = None
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        gains = []
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            gains.append(total)
                        
                        if gains:
                            avg_gain = np.mean(gains)
                            if avg_gain > max_gain:
                                max_gain = avg_gain
                                best_config = f"{grid}, {agents}ag"
            
            best_configs.append((interval, max_gain, best_config))
        
        intervals_str = [f"{int(i)}" if i != float('inf') else "∞" for i, _, _ in best_configs]
        gains = [g for _, g, _ in best_configs]
        
        bars = ax3.bar(range(len(intervals_str)), gains, alpha=0.8)
        ax3.set_xticks(range(len(intervals_str)))
        ax3.set_xticklabels(intervals_str)
        ax3.set_xlabel('Merge Interval')
        ax3.set_ylabel('Maximum Info Gain')
        ax3.set_title('Best Configuration per Interval')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add configuration labels
        for i, (_, gain, config) in enumerate(best_configs):
            ax3.text(i, gain + 0.1, config, ha='center', va='bottom', 
                   fontsize=8, rotation=45)
        
        # 4. Line plot: Interval effect across different problem sizes
        ax4 = plt.subplot(2, 3, 4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(grid_sizes)))
        
        for i, grid in enumerate(grid_sizes):
            interval_gains = []
            
            for interval in merge_intervals:
                gains = []
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            gains.append(total)
                
                if gains:
                    interval_gains.append(np.mean(gains))
                else:
                    interval_gains.append(0)
            
            ax4.plot(range(len(merge_intervals)), interval_gains, 'o-', 
                   color=colors[i], linewidth=2, markersize=8, 
                   alpha=0.8, label=f'Grid {grid}')
        
        ax4.set_xticks(range(len(merge_intervals)))
        ax4.set_xticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        ax4.set_xlabel('Merge Interval')
        ax4.set_ylabel('Average Total Info Gain')
        ax4.set_title('Communication Frequency Effect by Grid Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Scatter: Problem complexity vs info gain
        ax5 = plt.subplot(2, 3, 5)
        
        complexity_values = []
        gain_values = []
        interval_colors = []
        
        color_map = plt.cm.viridis(np.linspace(0, 1, len(merge_intervals)))
        
        for k, interval in enumerate(merge_intervals):
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        # Parse grid size
                        if isinstance(grid, str) and 'x' in grid:
                            rows, cols = map(int, grid.split('x'))
                            grid_area = rows * cols
                        else:
                            grid_area = 100
                        
                        complexity = grid_area * agents
                        
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            complexity_values.append(complexity)
                            gain_values.append(total)
                            interval_colors.append(color_map[k])
        
        scatter = ax5.scatter(complexity_values, gain_values, c=interval_colors, 
                            alpha=0.6, s=50)
        ax5.set_xlabel('Problem Complexity (Grid Area × Agents)')
        ax5.set_ylabel('Total Info Gain')
        ax5.set_title('Complexity vs Information Gain')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar for intervals
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=0, vmax=len(merge_intervals)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax5)
        cbar.set_label('Merge Interval')
        cbar.set_ticks(range(len(merge_intervals)))
        cbar.set_ticklabels([f"{int(i)}" if i != float('inf') else "∞" for i in merge_intervals])
        
        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_stats = []
        for interval in merge_intervals:
            all_gains = []
            
            for grid in grid_sizes:
                for agents in agent_numbers:
                    if (grid in time_series_data and 
                        agents in time_series_data[grid] and 
                        interval in time_series_data[grid][agents]):
                        
                        series_list = time_series_data[grid][agents][interval]['kl_series']
                        
                        for kl_series in series_list:
                            total = sum(max(0, kl_series[s-1] - kl_series[s]) 
                                      for s in range(1, len(kl_series)))
                            all_gains.append(total)
            
            if all_gains:
                int_str = f"{int(interval)}" if interval != float('inf') else "∞"
                summary_stats.append([
                    int_str,
                    f"{np.mean(all_gains):.3f}",
                    f"{np.std(all_gains):.3f}",
                    f"{np.min(all_gains):.3f}",
                    f"{np.max(all_gains):.3f}",
                    f"{len(all_gains)}"
                ])
        
        if summary_stats:
            table = ax6.table(
                cellText=summary_stats,
                colLabels=['Interval', 'Mean', 'Std', 'Min', 'Max', 'N'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0.2, 1, 0.6]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Highlight best interval
            best_idx = np.argmax([float(row[1]) for row in summary_stats]) + 1
            for j in range(len(summary_stats[0])):
                table[(best_idx, j)].set_facecolor('lightgreen')
                table[(best_idx, j)].set_alpha(0.5)
        
        ax6.set_title('Cumulative Info Gain Statistics', fontsize=12, fontweight='bold', y=0.9)
        
        plt.suptitle('Cumulative Information Gain: Comprehensive Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / 'cumulative_info_gain_comparison_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative comparison matrix saved: {output_path}")
    
    def run_analysis(self):
        """Run complete comprehensive analysis"""
        print(f"\nStarting comprehensive KL analysis...")
        print(f"="*60)
        
        if self.df.empty:
            print("No valid data found in results file!")
            return
        
        # Generate all analyses
        self.create_comprehensive_analysis()
        self.create_kl_evolution_analysis()  # Add this new analysis
        self.create_information_tracking_analysis()  # NEW: Add information tracking
        self.create_cumulative_information_gain_analysis()  # Add new analysis

        print(f"\nComprehensive analysis complete!")
        print(f"Generated files in {self.output_dir}:")
        print(f"  - grid_size_comprehensive_analysis.png")
        print(f"  - agent_count_comprehensive_analysis.png") 
        print(f"  - merge_interval_comprehensive_analysis.png")
        print(f"  - three_way_interaction_analysis.png")
        print(f"  - performance_surfaces.png")
        print(f"  - statistical_analysis.png")
        print(f"  - kl_evolution_by_interval.png")
        print(f"  - kl_evolution_by_grid_size.png")
        print(f"  - kl_evolution_by_agent_count.png")
        print(f"  - kl_evolution_overview.png")
        print(f"  - information_tracking_analysis.png")  # NEW
        print(f"  - cumulative_info_gain_by_interval.png")
        print(f"  - cumulative_info_gain_by_grid.png")
        print(f"  - cumulative_info_gain_by_agents.png")
        print(f"  - cumulative_info_gain_comparison_matrix.png")
        
        # Print key findings
        print(f"\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print(f"="*60)
        
        # Find overall best configuration
        best_idx = self.df['avg_kl_to_truth'].idxmin()
        best_config = self.df.loc[best_idx]
        
        print(f"Best overall performance:")
        print(f"  Grid: {best_config['grid_size']}")
        print(f"  Agents: {best_config['n_agents']}")
        print(f"  Interval: {int(best_config['merge_interval']) if best_config['merge_interval'] != float('inf') else '∞'}")
        print(f"  KL Divergence: {best_config['avg_kl_to_truth']:.4f}")
        
        # Performance by factor
        print(f"\nFactor-wise performance:")
        print(f"  Best grid size: {self.df.groupby('grid_size')['avg_kl_to_truth'].mean().idxmin()}")
        print(f"  Best agent count: {self.df.groupby('n_agents')['avg_kl_to_truth'].mean().idxmin()}")
        best_interval = self.df.groupby('merge_interval')['avg_kl_to_truth'].mean().idxmin()
        print(f"  Best merge interval: {int(best_interval) if best_interval != float('inf') else '∞'}")
        
        print(f"\nRange of performance:")
        print(f"  Best: {self.df['avg_kl_to_truth'].min():.4f}")
        print(f"  Worst: {self.df['avg_kl_to_truth'].max():.4f}")
        print(f"  Improvement potential: {(self.df['avg_kl_to_truth'].max() - self.df['avg_kl_to_truth'].min()) / self.df['avg_kl_to_truth'].max() * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive KL-divergence performance analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
                Examples:
                python comprehensive_kl_analysis.py results/consolidated_results_20241219_143022.pkl
                python comprehensive_kl_analysis.py results/kl_focused_results_20241219_150000.pkl
                        '''
        )
    
    parser.add_argument('results_file', 
                       help='Path to the pickle file containing results')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"ERROR: Results file not found: {args.results_file}")
        return 1
    
    try:
        analyzer = ComprehensiveKLAnalyzer(args.results_file)
        analyzer.run_analysis()
        return 0
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())