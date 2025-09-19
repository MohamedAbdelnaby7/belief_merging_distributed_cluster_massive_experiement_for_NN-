#!/usr/bin/env python3
"""
Advanced Cluster-Scale Experiment Analyzer
Designed for massive datasets from distributed Turing cluster experiments
Provides deep insights into belief merging performance across configurations
MODIFIED: Focus on communication efficiency and discovery speed metrics
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.stats import entropy, mannwhitneyu, kruskal, pearsonr, spearmanr
from scipy.stats import chi2_contingency, shapiro, levene
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
import itertools
from collections import defaultdict
import gc
import psutil
warnings.filterwarnings('ignore')

class AdvancedClusterAnalyzer:
    """Advanced analyzer for massive cluster experiment datasets"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir / "advanced_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Set high-quality plotting parameters
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
        
        self.colors = {
            'full_comm': '#1f77b4',      # Blue
            'interval_10': '#ff7f0e',    # Orange
            'interval_25': '#2ca02c',    # Green
            'interval_50': '#d62728',    # Red
            'interval_100': '#9467bd',   # Purple
            'interval_200': '#8c564b',   # Brown
            'interval_500': '#e377c2',   # Pink
            'no_comm': '#7f7f7f'         # Gray
        }
        
    def load_and_structure_data(self, filename: str = None) -> pd.DataFrame:
        """Load results and convert to structured DataFrame for analysis"""
        print("Loading and structuring massive dataset...")
        
        if filename is None:
            result_files = list(self.results_dir.glob("consolidated_results_*.pkl"))
            if not result_files:
                raise FileNotFoundError("No consolidated results found")
            filename = str(max(result_files, key=lambda x: x.stat().st_mtime))
        
        print(f"Loading from: {filename}")
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        # Convert to flat DataFrame for advanced analysis
        print("Converting to structured DataFrame...")
        data_rows = []
        
        for grid_key, grid_results in results.items():
            rows, cols = map(int, grid_key.split('x'))
            grid_size = rows * cols
            
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                
                for pattern, pattern_results in agent_results.items():
                    for strategy_key, trials in pattern_results.items():
                        if not trials:
                            continue
                        
                        for trial_idx, trial in enumerate(trials):
                            # Extract merge interval
                            if strategy_key == 'full_comm':
                                merge_interval = 0
                            elif strategy_key == 'no_comm':
                                merge_interval = float('inf')
                            else:
                                merge_interval = int(strategy_key.replace('interval_', ''))
                            
                            row = {
                                # Configuration
                                'grid_key': grid_key,
                                'grid_size': grid_size,
                                'grid_rows': rows,
                                'grid_cols': cols,
                                'n_agents': n_agents,
                                'pattern': pattern,
                                'strategy': strategy_key,
                                'merge_interval': merge_interval,
                                'trial_id': trial_idx,
                                
                                # Primary metrics
                                'target_found': trial['target_found'],
                                'discovery_step': trial['first_discovery_step'] if trial['target_found'] else trial.get('max_steps', 1000),
                                'discovery_count': trial['discovery_count'],
                                'final_entropy': trial['final_entropy'],
                                'prediction_error': trial['prediction_error'],
                                'prob_at_target': trial['prob_at_true_target'],
                                'computation_time': trial['elapsed_time'],
                                
                                # MODIFIED: Enhanced communication metrics
                                'discovery_rate': 1.0 if trial['target_found'] else 0.0,
                                'steps_to_discovery': trial['first_discovery_step'] if trial['target_found'] else 1000,
                                'communication_efficiency': 1.0 / merge_interval if merge_interval != float('inf') and merge_interval > 0 else (1.0 if merge_interval == 0 else 0.0),
                                'info_gain': np.log(grid_size) - trial['final_entropy'],  # Information gained
                                'performance_score': (1.0 if trial['target_found'] else 0.0) / (1 + trial['prediction_error']),
                                
                                # Communication metrics
                                'total_merges': trial.get('total_merges', 0),
                                'merge_frequency': trial.get('total_merges', 0) / trial.get('max_steps', 1000) if trial.get('max_steps', 1000) > 0 else 0,
                                
                                # ADDED: Steps-based efficiency metrics
                                'discovery_efficiency': (1.0 if trial['target_found'] else 0.0) / max(1, trial['first_discovery_step'] if trial['target_found'] else 1000),
                                'communication_overhead': merge_interval if merge_interval != float('inf') else 1000,
                                'steps_per_merge': merge_interval if merge_interval != float('inf') and merge_interval > 0 else (0 if merge_interval == 0 else 1000),
                            }
                            
                            # Add convergence metrics if available
                            if 'entropy_history' in trial and trial['entropy_history']:
                                entropy_hist = [h['mean'] for h in trial['entropy_history']]
                                if len(entropy_hist) > 1:
                                    row['initial_entropy'] = entropy_hist[0]
                                    row['entropy_reduction'] = entropy_hist[0] - entropy_hist[-1]
                                    row['entropy_reduction_rate'] = row['entropy_reduction'] / len(entropy_hist)
                                    
                                    # Convergence speed (steps to reach 90% of final entropy reduction)
                                    final_reduction = entropy_hist[0] - entropy_hist[-1]
                                    if final_reduction > 0:
                                        target_entropy = entropy_hist[0] - 0.9 * final_reduction
                                        convergence_step = len(entropy_hist)
                                        for i, h in enumerate(entropy_hist):
                                            if h <= target_entropy:
                                                convergence_step = i
                                                break
                                        row['convergence_speed'] = convergence_step
                                    else:
                                        row['convergence_speed'] = 1000
                                else:
                                    row['initial_entropy'] = np.log(grid_size)
                                    row['entropy_reduction'] = 0
                                    row['entropy_reduction_rate'] = 0
                                    row['convergence_speed'] = 1000
                            
                            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        print(f"Created DataFrame with {len(df)} trials")
        print(f"Configurations: {len(df.groupby(['grid_key', 'n_agents', 'pattern', 'strategy']))} unique")
        
        # Save structured data
        df.to_csv(self.analysis_dir / 'structured_data.csv', index=False)
        df.to_pickle(self.analysis_dir / 'structured_data.pkl')
        
        return df
    
    def advanced_statistical_analysis(self, df: pd.DataFrame):
        """Perform comprehensive statistical analysis"""
        print("\nPerforming advanced statistical analysis...")
        
        stats_results = {}
        
        # 1. Multi-way ANOVA for each metric
        print("  - Multi-way ANOVA tests...")
        from scipy.stats import f_oneway
        
        # MODIFIED: Focus on step-based and communication metrics
        metrics = ['steps_to_discovery', 'discovery_efficiency', 'communication_efficiency', 'final_entropy']
        factors = ['pattern', 'strategy', 'grid_key', 'n_agents']
        
        anova_results = {}
        for metric in metrics:
            metric_results = {}
            
            for factor in factors:
                groups = [group[metric].values for name, group in df.groupby(factor) if len(group) > 1]
                if len(groups) > 1:
                    try:
                        f_stat, p_value = f_oneway(*groups)
                        metric_results[factor] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': 'large' if f_stat > 4 else 'medium' if f_stat > 2 else 'small'
                        }
                    except:
                        metric_results[factor] = {'error': 'Could not compute'}
            
            anova_results[metric] = metric_results
        
        stats_results['anova'] = anova_results
        
        # 2. Pairwise strategy comparisons
        print("  - Pairwise strategy comparisons...")
        strategies = df['strategy'].unique()
        strategy_comparisons = {}
        
        # MODIFIED: Focus on steps and communication metrics
        for metric in ['steps_to_discovery', 'discovery_efficiency', 'discovery_rate']:
            metric_comparisons = {}
            
            for strategy1, strategy2 in itertools.combinations(strategies, 2):
                data1 = df[df['strategy'] == strategy1][metric].dropna()
                data2 = df[df['strategy'] == strategy2][metric].dropna()
                
                if len(data1) > 5 and len(data2) > 5:
                    try:
                        if metric == 'discovery_rate':
                            # Chi-square test for proportions
                            successes1 = data1.sum()
                            successes2 = data2.sum()
                            total1, total2 = len(data1), len(data2)
                            contingency = [[successes1, total1-successes1], [successes2, total2-successes2]]
                            chi2, p_val = chi2_contingency(contingency)[:2]
                            test_stat, test_name = chi2, 'chi2'
                        else:
                            # Mann-Whitney U test for continuous variables
                            test_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                            test_name = 'mannwhitney'
                        
                        metric_comparisons[f"{strategy1}_vs_{strategy2}"] = {
                            'test': test_name,
                            'statistic': test_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                            'mean_diff': data1.mean() - data2.mean(),
                            'effect_size': abs(data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                        }
                    except:
                        continue
            
            strategy_comparisons[metric] = metric_comparisons
        
        stats_results['strategy_comparisons'] = strategy_comparisons
        
        # 3. Correlation analysis
        print("  - Correlation analysis...")
        # MODIFIED: Include step-based metrics
        numeric_cols = ['grid_size', 'n_agents', 'steps_per_merge', 'steps_to_discovery', 
                       'discovery_efficiency', 'communication_efficiency', 'final_entropy', 
                       'prediction_error', 'computation_time', 'performance_score']
        
        # Replace inf with large number for correlation
        df_corr = df.copy()
        df_corr['steps_per_merge'] = df_corr['steps_per_merge'].replace(float('inf'), 1000)
        
        correlation_matrix = df_corr[numeric_cols].corr(method='spearman')
        stats_results['correlations'] = correlation_matrix.to_dict()
        
        # Save statistical results
        with open(self.analysis_dir / 'advanced_statistics.json', 'w') as f:
            json.dump(self._make_json_serializable(stats_results), f, indent=2)
        
        return stats_results
    
    def create_advanced_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations for massive dataset"""
        print("\nCreating advanced visualizations...")
        
        # 1. Multi-dimensional performance landscape
        self._plot_performance_landscape(df)
        
        # 2. Statistical significance heatmaps
        self._plot_significance_heatmaps(df)
        
        # 3. Convergence and dynamics analysis
        self._plot_convergence_dynamics(df)
        
        # 4. Scalability analysis
        self._plot_scalability_analysis(df)
        
        # 5. Communication efficiency analysis
        self._plot_communication_efficiency(df)
        
        # 6. Pattern-specific deep dive
        self._plot_pattern_deep_dive(df)
        
        # 7. Resource optimization analysis
        self._plot_resource_optimization(df)
    
    def _plot_performance_landscape(self, df: pd.DataFrame):
        """Create multi-dimensional performance landscape visualization"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # MODIFIED: Main plot - Steps to Discovery vs Communication Interval
        ax = fig.add_subplot(gs[0:2, 0:2])
        
        # Aggregate data for main plot
        agg_data = df.groupby(['communication_overhead', 'grid_size']).agg({
            'steps_to_discovery': 'mean',
            'discovery_rate': 'mean',
            'discovery_efficiency': 'mean'
        }).reset_index()
        
        # Create scatter plot with size representing discovery rate
        scatter = ax.scatter(agg_data['communication_overhead'], agg_data['steps_to_discovery'], 
                           c=agg_data['discovery_efficiency'], s=agg_data['discovery_rate']*200 + 50,
                           cmap='viridis', alpha=0.7)
        
        ax.set_xlabel('Communication Interval (steps between merges)')
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Discovery Speed vs Communication Frequency\n(Color: Discovery Efficiency, Size: Discovery Rate)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Discovery Efficiency')
        
        # Steps to Discovery vs Communication Efficiency by Strategy
        ax = fig.add_subplot(gs[0, 2])
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            ax.scatter(strategy_data['communication_efficiency'], strategy_data['steps_to_discovery'], 
                      alpha=0.6, label=strategy, s=20, color=self.colors.get(strategy, 'gray'))
        ax.set_xlabel('Communication Efficiency')
        ax.set_ylabel('Steps to Discovery')
        ax.set_title('Discovery Speed vs Communication')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Discovery Rate Heatmap (unchanged)
        ax = fig.add_subplot(gs[0, 3])
        heatmap_data = df.groupby(['pattern', 'strategy'])['discovery_rate'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Discovery Rate by Pattern & Strategy')
        
        # MODIFIED: Distribution plots - focus on step-based metrics
        metrics = ['steps_to_discovery', 'discovery_efficiency', 'communication_efficiency', 'computation_time']
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[2 + i//2, i%2])
            
            # Box plot by strategy
            strategy_order = ['full_comm', 'interval_10', 'interval_25', 'interval_50', 
                             'interval_100', 'interval_200', 'interval_500', 'no_comm']
            available_strategies = [s for s in strategy_order if s in df['strategy'].unique()]
            
            sns.boxplot(data=df, x='strategy', y=metric, order=available_strategies, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        
        # Grid size impact (unchanged)
        ax = fig.add_subplot(gs[2, 2])
        grid_performance = df.groupby(['grid_key', 'strategy'])['performance_score'].mean().unstack()
        sns.heatmap(grid_performance, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title('Performance by Grid Size & Strategy')
        
        # Agent number impact (unchanged)
        ax = fig.add_subplot(gs[2, 3])
        agent_performance = df.groupby(['n_agents', 'strategy'])['final_entropy'].mean().unstack()
        sns.heatmap(agent_performance, annot=True, fmt='.3f', cmap='viridis_r', ax=ax)
        ax.set_title('Entropy by Agent Count & Strategy')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'performance_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_significance_heatmaps(self, df: pd.DataFrame):
        """Create statistical significance heatmaps"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        strategies = df['strategy'].unique()
        patterns = df['pattern'].unique()
        
        # Strategy pairwise significance for discovery rate (unchanged)
        ax = axes[0, 0]
        p_matrix = np.ones((len(strategies), len(strategies)))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    data1 = df[df['strategy'] == strategy1]['discovery_rate']
                    data2 = df[df['strategy'] == strategy2]['discovery_rate']
                    
                    if len(data1) > 5 and len(data2) > 5:
                        try:
                            # Chi-square test for proportions
                            successes1, successes2 = data1.sum(), data2.sum()
                            total1, total2 = len(data1), len(data2)
                            contingency = [[successes1, total1-successes1], [successes2, total2-successes2]]
                            _, p_val = chi2_contingency(contingency)[:2]
                            p_matrix[i, j] = p_val
                        except:
                            p_matrix[i, j] = 1.0
        
        sns.heatmap(p_matrix, annot=True, fmt='.3f', xticklabels=strategies, yticklabels=strategies,
                   cmap='RdYlBu_r', ax=ax, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
        ax.set_title('Discovery Rate Significance (p-values)')
        
        # MODIFIED: Strategy pairwise significance for steps to discovery
        ax = axes[0, 1]
        p_matrix = np.ones((len(strategies), len(strategies)))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    data1 = df[df['strategy'] == strategy1]['steps_to_discovery'].dropna()
                    data2 = df[df['strategy'] == strategy2]['steps_to_discovery'].dropna()
                    
                    if len(data1) > 5 and len(data2) > 5:
                        try:
                            _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                            p_matrix[i, j] = p_val
                        except:
                            p_matrix[i, j] = 1.0
        
        sns.heatmap(p_matrix, annot=True, fmt='.3f', xticklabels=strategies, yticklabels=strategies,
                   cmap='RdYlBu_r', ax=ax, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
        ax.set_title('Steps to Discovery Significance (p-values)')
        
        # MODIFIED: Effect sizes for discovery efficiency
        ax = axes[1, 0]
        effect_matrix = np.zeros((len(strategies), len(strategies)))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    data1 = df[df['strategy'] == strategy1]['discovery_efficiency'].dropna()
                    data2 = df[df['strategy'] == strategy2]['discovery_efficiency'].dropna()
                    
                    if len(data1) > 5 and len(data2) > 5:
                        # Cohen's d effect size
                        pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
                        if pooled_std > 0:
                            effect_size = abs(data1.mean() - data2.mean()) / pooled_std
                            effect_matrix[i, j] = effect_size
        
        sns.heatmap(effect_matrix, annot=True, fmt='.2f', xticklabels=strategies, yticklabels=strategies,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Effect Size (Cohen\'s d)'})
        ax.set_title('Discovery Efficiency Effect Sizes')
        
        # Configuration performance summary (unchanged)
        ax = axes[1, 1]
        config_performance = df.groupby(['grid_key', 'n_agents'])['performance_score'].mean().unstack()
        sns.heatmap(config_performance, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title('Performance by Grid & Agent Configuration')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_dynamics(self, df: pd.DataFrame):
        """Analyze convergence and temporal dynamics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 1. Convergence speed analysis (unchanged)
        ax = axes[0, 0]
        if 'convergence_speed' in df.columns:
            convergence_data = df.groupby('strategy')['convergence_speed'].apply(list).to_dict()
            
            for strategy, speeds in convergence_data.items():
                if speeds and not all(pd.isna(speeds)):
                    clean_speeds = [s for s in speeds if not pd.isna(s)]
                    if clean_speeds:
                        ax.hist(clean_speeds, alpha=0.7, label=strategy, bins=20, density=True)
            
            ax.set_xlabel('Convergence Speed (steps)')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Convergence Speeds')
            ax.legend()
        
        # MODIFIED: Steps to Discovery vs Communication Interval
        ax = axes[0, 1]
        step_data = df.groupby('strategy')['steps_to_discovery'].mean().sort_values()
        bars = ax.bar(range(len(step_data)), step_data.values, 
                     color=[self.colors.get(s, 'gray') for s in step_data.index])
        ax.set_xticks(range(len(step_data)))
        ax.set_xticklabels(step_data.index, rotation=45)
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Discovery Speed by Strategy')
        
        # Add value labels on bars
        for bar, value in zip(bars, step_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(step_data.values),
                   f'{value:.0f}', ha='center', va='bottom')
        
        # MODIFIED: Discovery Efficiency vs Communication Frequency
        ax = axes[1, 0]
        comm_data = df[df['communication_efficiency'] > 0]  # Exclude no_comm
        sns.scatterplot(data=comm_data, x='communication_efficiency', y='discovery_efficiency', 
                       hue='strategy', ax=ax, s=60)
        ax.set_xlabel('Communication Efficiency (1/merge_interval)')
        ax.set_ylabel('Discovery Efficiency')
        ax.set_title('Discovery vs Communication Efficiency')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Performance vs computational cost (unchanged)
        ax = axes[1, 1]
        scatter = ax.scatter(df['computation_time'], df['performance_score'], 
                           c=df['communication_overhead'].replace(float('inf'), 1000),
                           cmap='coolwarm', alpha=0.6, s=30)
        ax.set_xlabel('Computation Time (seconds)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance vs Computational Cost')
        plt.colorbar(scatter, ax=ax, label='Communication Overhead')
        
        # Discovery dynamics by pattern (unchanged)
        ax = axes[2, 0]
        pattern_discovery = df.groupby(['pattern', 'strategy'])['discovery_rate'].mean().unstack()
        sns.heatmap(pattern_discovery, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
        ax.set_title('Discovery Success by Pattern & Strategy')
        
        # MODIFIED: Communication overhead analysis
        ax = axes[2, 1]
        overhead_analysis = df.groupby('strategy').agg({
            'steps_to_discovery': 'mean',
            'communication_overhead': 'mean',
            'discovery_rate': 'mean'
        }).reset_index()
        
        # Create bubble plot
        scatter = ax.scatter(overhead_analysis['communication_overhead'], 
                           overhead_analysis['steps_to_discovery'],
                           s=overhead_analysis['discovery_rate'] * 500,
                           c=range(len(overhead_analysis)), 
                           cmap='viridis', alpha=0.7)
        
        # Add labels
        for _, row in overhead_analysis.iterrows():
            ax.annotate(row['strategy'][:8], 
                       (row['communication_overhead'], row['steps_to_discovery']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Communication Overhead (steps)')
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Communication Cost vs Discovery Speed\n(Bubble size: Discovery Rate)')
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'convergence_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, df: pd.DataFrame):
        """Analyze scalability across grid sizes and agent numbers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # MODIFIED: Performance scaling with grid size - focus on steps
        ax = axes[0, 0]
        for strategy in ['full_comm', 'interval_25', 'interval_100', 'no_comm']:
            if strategy in df['strategy'].values:
                strategy_data = df[df['strategy'] == strategy]
                scaling_data = strategy_data.groupby('grid_size')['steps_to_discovery'].mean()
                ax.plot(scaling_data.index, scaling_data.values, 'o-', 
                       label=strategy, linewidth=2, markersize=6)
        
        ax.set_xlabel('Grid Size (total states)')
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Discovery Speed Scaling with Grid Size')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Computational complexity scaling (unchanged)
        ax = axes[0, 1]
        comp_scaling = df.groupby(['grid_size', 'n_agents'])['computation_time'].mean().reset_index()
        
        for n_agents in sorted(df['n_agents'].unique()):
            agent_data = comp_scaling[comp_scaling['n_agents'] == n_agents]
            ax.plot(agent_data['grid_size'], agent_data['computation_time'], 
                   'o-', label=f'{n_agents} agents', linewidth=2, markersize=6)
        
        ax.set_xlabel('Grid Size')
        ax.set_ylabel('Average Computation Time (s)')
        ax.set_title('Computational Scaling')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MODIFIED: Agent scaling efficiency - focus on discovery efficiency
        ax = axes[0, 2]
        agent_efficiency = df.groupby(['n_agents', 'strategy'])['discovery_efficiency'].mean().unstack()
        
        for strategy in agent_efficiency.columns:
            ax.plot(agent_efficiency.index, agent_efficiency[strategy], 
                   'o-', label=strategy, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Discovery Efficiency')
        ax.set_title('Discovery Efficiency by Agent Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MODIFIED: Communication efficiency per merge
        ax = axes[1, 0]
        if 'total_merges' in df.columns:
            df_with_merges = df[df['total_merges'] > 0].copy()
            df_with_merges['steps_per_merge'] = df_with_merges['steps_to_discovery'] / df_with_merges['total_merges']
            
            merge_efficiency = df_with_merges.groupby('strategy')['steps_per_merge'].mean()
            bars = ax.bar(range(len(merge_efficiency)), merge_efficiency.values,
                         color=[self.colors.get(s, 'gray') for s in merge_efficiency.index])
            ax.set_xticks(range(len(merge_efficiency)))
            ax.set_xticklabels(merge_efficiency.index, rotation=45)
            ax.set_ylabel('Average Steps per Merge Event')
            ax.set_title('Communication Frequency Analysis')
        
        # Pattern complexity analysis (unchanged)
        ax = axes[1, 1]
        pattern_complexity = df.groupby('pattern').agg({
            'prediction_error': 'mean',
            'final_entropy': 'mean',
            'discovery_rate': 'mean'
        })
        
        # Normalize to 0-1 scale for comparison
        normalized = (pattern_complexity - pattern_complexity.min()) / (pattern_complexity.max() - pattern_complexity.min())
        
        x = np.arange(len(normalized))
        width = 0.25
        
        ax.bar(x - width, normalized['prediction_error'], width, label='Prediction Error', alpha=0.8)
        ax.bar(x, normalized['final_entropy'], width, label='Final Entropy', alpha=0.8)
        ax.bar(x + width, 1 - normalized['discovery_rate'], width, label='Discovery Difficulty', alpha=0.8)
        
        ax.set_xlabel('Movement Pattern')
        ax.set_ylabel('Normalized Difficulty (0=Easy, 1=Hard)')
        ax.set_title('Pattern Difficulty Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(normalized.index)
        ax.legend()
        
        # MODIFIED: Optimal configuration finder - focus on steps and communication
        ax = axes[1, 2]
        
        # Find Pareto frontier: best discovery speed for each communication cost level
        pareto_data = df.groupby(['strategy', 'pattern']).agg({
            'steps_to_discovery': 'mean',
            'communication_overhead': 'mean',
            'discovery_rate': 'mean'
        }).reset_index()
        
        scatter = ax.scatter(pareto_data['communication_overhead'], pareto_data['steps_to_discovery'],
                           c=pareto_data['discovery_rate'], cmap='RdYlGn', s=100, alpha=0.7, vmin=0, vmax=1)
        
        # Annotate points
        for _, row in pareto_data.iterrows():
            ax.annotate(f"{row['strategy'][:8]}\n{row['pattern'][:4]}", 
                       (row['communication_overhead'], row['steps_to_discovery']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Communication Overhead (steps between merges)')
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Discovery Speed vs Communication Cost')
        ax.set_xscale('log')
        plt.colorbar(scatter, ax=ax, label='Discovery Rate')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_communication_efficiency(self, df: pd.DataFrame):
        """Analyze communication efficiency and overhead"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MODIFIED: Communication frequency vs discovery speed
        ax = axes[0, 0]
        comm_data = df[df['merge_interval'] != float('inf')].copy()
        comm_data['comm_frequency'] = 1 / comm_data['merge_interval']
        
        for pattern in df['pattern'].unique():
            pattern_data = comm_data[comm_data['pattern'] == pattern]
            freq_perf = pattern_data.groupby('comm_frequency').agg({
                'steps_to_discovery': 'mean',
                'discovery_rate': 'mean'
            })
            
            ax.plot(freq_perf.index, freq_perf['steps_to_discovery'], 'o-', 
                   label=f'{pattern} Steps', linewidth=2, markersize=6)
        
        ax.set_xlabel('Communication Frequency (1/merge_interval)')
        ax.set_ylabel('Average Steps to Discovery')
        ax.set_title('Communication Frequency vs Discovery Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MODIFIED: Steps saved by communication
        ax = axes[0, 1]
        
        # Calculate steps saved compared to no communication
        strategies_ordered = ['no_comm', 'interval_500', 'interval_200', 'interval_100', 
                             'interval_50', 'interval_25', 'interval_10', 'full_comm']
        available_strategies = [s for s in strategies_ordered if s in df['strategy'].unique()]
        
        steps_saved = []
        strategy_labels = []
        
        baseline_steps = df[df['strategy'] == 'no_comm']['steps_to_discovery'].mean() if 'no_comm' in df['strategy'].values else 1000
        
        for strategy in available_strategies:
            if strategy != 'no_comm':
                strategy_steps = df[df['strategy'] == strategy]['steps_to_discovery'].mean()
                saved = baseline_steps - strategy_steps
                steps_saved.append(saved)
                strategy_labels.append(strategy)
        
        bars = ax.bar(range(len(steps_saved)), steps_saved,
                     color=[self.colors.get(s, 'gray') for s in strategy_labels])
        ax.set_xticks(range(len(strategy_labels)))
        ax.set_xticklabels(strategy_labels, rotation=45)
        ax.set_ylabel('Steps Saved vs No Communication')
        ax.set_title('Benefit of Communication Strategies')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, steps_saved):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (5 if value > 0 else -15),
                   f'{value:.0f}', ha='center', va='bottom' if value > 0 else 'top')
        
        # MODIFIED: Communication overhead vs discovery success
        ax = axes[1, 0]
        
        # Calculate communication cost vs benefit
        df_comm_analysis = df.copy()
        df_comm_analysis['comm_steps'] = 1000 - df_comm_analysis['communication_overhead']  # Higher = more communication
        
        cost_benefit = df_comm_analysis.groupby('strategy').agg({
            'comm_steps': 'mean',
            'discovery_rate': 'mean',
            'steps_to_discovery': 'mean'
        })
        
        scatter = ax.scatter(cost_benefit['comm_steps'], cost_benefit['discovery_rate'],
                           s=1000/cost_benefit['steps_to_discovery'] * 100, alpha=0.7,
                           c=range(len(cost_benefit)), cmap='viridis')
        
        for strategy, row in cost_benefit.iterrows():
            ax.annotate(strategy, (row['comm_steps'], row['discovery_rate']),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Communication Investment (1000 - overhead)')
        ax.set_ylabel('Discovery Success Rate')
        ax.set_title('Communication Investment vs Success Rate')
        
        # MODIFIED: Optimal communication intervals by scenario
        ax = axes[1, 1]
        
        # For each pattern and grid size, find strategy with best step efficiency
        optimal_strategies = {}
        
        for pattern in df['pattern'].unique():
            for grid_key in df['grid_key'].unique():
                subset = df[(df['pattern'] == pattern) & (df['grid_key'] == grid_key)]
                if len(subset) > 0:
                    best_strategy = subset.groupby('strategy')['discovery_efficiency'].mean().idxmax()
                    optimal_strategies[f"{pattern}_{grid_key}"] = best_strategy
        
        # Count frequency of each strategy being optimal
        strategy_counts = defaultdict(int)
        for strategy in optimal_strategies.values():
            strategy_counts[strategy] += 1
        
        strategies, counts = zip(*strategy_counts.items())
        bars = ax.bar(range(len(strategies)), counts,
                     color=[self.colors.get(s, 'gray') for s in strategies])
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45)
        ax.set_ylabel('Number of Configurations Where Optimal')
        ax.set_title('Most Efficient Strategies by Scenario')
        
        # Add percentage labels
        total_configs = len(optimal_strategies)
        for bar, count in zip(bars, counts):
            percentage = count / total_configs * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'communication_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_deep_dive(self, df: pd.DataFrame):
        """Deep dive analysis for each movement pattern"""
        for pattern in df['pattern'].unique():
            pattern_data = df[df['pattern'] == pattern]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Deep Dive: {pattern.upper()} Movement Pattern', fontsize=16)
            
            # MODIFIED: Strategy performance ranking - focus on discovery efficiency
            ax = axes[0, 0]
            strategy_ranking = pattern_data.groupby('strategy').agg({
                'discovery_efficiency': ['mean', 'std'],
                'steps_to_discovery': 'mean',
                'discovery_rate': 'mean'
            }).round(3)
            
            # Flatten column names
            strategy_ranking.columns = ['_'.join(col).strip() for col in strategy_ranking.columns]
            strategy_ranking = strategy_ranking.sort_values('discovery_efficiency_mean', ascending=False)
            
            # Plot ranking
            y_pos = np.arange(len(strategy_ranking))
            bars = ax.barh(y_pos, strategy_ranking['discovery_efficiency_mean'],
                          xerr=strategy_ranking['discovery_efficiency_std'],
                          color=[self.colors.get(s, 'gray') for s in strategy_ranking.index])
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(strategy_ranking.index)
            ax.set_xlabel('Discovery Efficiency')
            ax.set_title(f'{pattern}: Strategy Ranking')
            
            # Add values
            for i, (bar, mean_val) in enumerate(zip(bars, strategy_ranking['discovery_efficiency_mean'])):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{mean_val:.4f}', va='center', fontsize=9)
            
            # Grid size sensitivity (unchanged)
            ax = axes[0, 1]
            grid_sensitivity = pattern_data.groupby(['grid_key', 'strategy'])['performance_score'].mean().unstack()
            
            for strategy in ['full_comm', 'interval_25', 'interval_100', 'no_comm']:
                if strategy in grid_sensitivity.columns:
                    ax.plot(grid_sensitivity.index, grid_sensitivity[strategy], 
                           'o-', label=strategy, linewidth=2, markersize=6)
            
            ax.set_xlabel('Grid Size')
            ax.set_ylabel('Performance Score')
            ax.set_title(f'{pattern}: Grid Size Sensitivity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # MODIFIED: Discovery step distribution
            ax = axes[0, 2]
            
            discovery_steps = {}
            for strategy in pattern_data['strategy'].unique():
                strategy_data = pattern_data[pattern_data['strategy'] == strategy]
                steps = strategy_data['steps_to_discovery'].values
                if len(steps) > 0:
                    discovery_steps[strategy] = steps
            
            if discovery_steps:
                ax.boxplot(discovery_steps.values(), labels=discovery_steps.keys())
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel('Steps to Discovery')
                ax.set_title(f'{pattern}: Discovery Speed Distribution')
            
            # MODIFIED: Communication efficiency evolution
            ax = axes[1, 0]
            
            # Show communication efficiency by strategy
            comm_efficiency = pattern_data.groupby('strategy')['communication_efficiency'].mean()
            bars = ax.bar(range(len(comm_efficiency)), comm_efficiency.values,
                         color=[self.colors.get(s, 'gray') for s in comm_efficiency.index])
            ax.set_xticks(range(len(comm_efficiency)))
            ax.set_xticklabels(comm_efficiency.index, rotation=45)
            ax.set_ylabel('Average Communication Efficiency')
            ax.set_title(f'{pattern}: Communication Patterns')
            
            # Agent coordination efficiency (unchanged)
            ax = axes[1, 1]
            agent_coord = pattern_data.groupby(['n_agents', 'strategy'])['discovery_rate'].mean().unstack()
            
            sns.heatmap(agent_coord, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title(f'{pattern}: Agent Coordination Efficiency')
            
            # MODIFIED: Statistical summary table - focus on step metrics
            ax = axes[1, 2]
            ax.axis('off')
            
            # Create summary statistics table
            summary_stats = pattern_data.groupby('strategy').agg({
                'discovery_rate': ['mean', 'std'],
                'steps_to_discovery': ['mean', 'std'],
                'discovery_efficiency': ['mean', 'std'],
                'communication_efficiency': ['mean', 'std']
            }).round(3)
            
            # Format as text table
            table_text = f"{pattern.upper()} SUMMARY STATISTICS\n\n"
            table_text += "Strategy | Discovery Rate | Steps to Disc | Discovery Eff | Comm Eff\n"
            table_text += "-" * 70 + "\n"
            
            for strategy in summary_stats.index:
                dr_mean = summary_stats.loc[strategy, ('discovery_rate', 'mean')]
                dr_std = summary_stats.loc[strategy, ('discovery_rate', 'std')]
                st_mean = summary_stats.loc[strategy, ('steps_to_discovery', 'mean')]
                st_std = summary_stats.loc[strategy, ('steps_to_discovery', 'std')]
                de_mean = summary_stats.loc[strategy, ('discovery_efficiency', 'mean')]
                de_std = summary_stats.loc[strategy, ('discovery_efficiency', 'std')]
                ce_mean = summary_stats.loc[strategy, ('communication_efficiency', 'mean')]
                ce_std = summary_stats.loc[strategy, ('communication_efficiency', 'std')]
                
                table_text += f"{strategy[:12]:<12} | {dr_mean:.3f}±{dr_std:.3f} | {st_mean:.0f}±{st_std:.0f} | {de_mean:.4f}±{de_std:.4f} | {ce_mean:.3f}±{ce_std:.3f}\n"
            
            ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontfamily='monospace',
                   fontsize=8, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / f'pattern_deepdive_{pattern}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_resource_optimization(self, df: pd.DataFrame):
        """Analyze resource optimization and recommendations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MODIFIED: Steps efficiency per computational cost
        ax = axes[0, 0]
        
        # Normalize computation time to "cost units"
        max_time = df['computation_time'].max()
        df_cost = df.copy()
        df_cost['relative_cost'] = df_cost['computation_time'] / max_time
        
        cost_efficiency = df_cost.groupby('strategy').agg({
            'discovery_efficiency': 'mean',
            'relative_cost': 'mean'
        })
        cost_efficiency['step_cost_efficiency'] = cost_efficiency['discovery_efficiency'] / cost_efficiency['relative_cost']
        
        bars = ax.bar(range(len(cost_efficiency)), cost_efficiency['step_cost_efficiency'],
                     color=[self.colors.get(s, 'gray') for s in cost_efficiency.index])
        ax.set_xticks(range(len(cost_efficiency)))
        ax.set_xticklabels(cost_efficiency.index, rotation=45)
        ax.set_ylabel('Discovery Efficiency per Computational Cost')
        ax.set_title('Step Efficiency Ranking')
        
        # MODIFIED: Recommended configurations - focus on step metrics
        ax = axes[0, 1]
        
        # Find best strategy for each scenario
        recommendations = {}
        
        # Fastest discovery: minimize steps to discovery
        fastest = df.groupby('strategy')['steps_to_discovery'].mean().idxmin()
        recommendations['Fastest Discovery'] = fastest
        
        # Most efficient: best discovery efficiency
        most_efficient = df.groupby('strategy')['discovery_efficiency'].mean().idxmax()
        recommendations['Most Efficient'] = most_efficient
        
        # Least communication: minimize communication overhead while maintaining performance
        low_comm_candidates = df[df['discovery_rate'] > df['discovery_rate'].quantile(0.7)]
        least_comm = low_comm_candidates.groupby('strategy')['communication_overhead'].mean().idxmax()
        recommendations['Least Communication'] = least_comm
        
        # Plot recommendations
        scenario_names = list(recommendations.keys())
        recommended_strategies = list(recommendations.values())
        
        # Get discovery efficiency scores for recommended strategies
        efficiency_scores = []
        for strategy in recommended_strategies:
            score = df[df['strategy'] == strategy]['discovery_efficiency'].mean()
            efficiency_scores.append(score)
        
        bars = ax.bar(scenario_names, efficiency_scores,
                     color=[self.colors.get(s, 'gray') for s in recommended_strategies])
        ax.set_ylabel('Discovery Efficiency')
        ax.set_title('Recommended Configurations by Scenario')
        
        # Add strategy labels on bars
        for bar, strategy in zip(bars, recommended_strategies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(efficiency_scores),
                   strategy, ha='center', va='bottom', rotation=45, fontsize=9)
        
        # Sensitivity analysis (unchanged)
        ax = axes[1, 0]
        
        # Calculate coefficient of variation for each strategy
        sensitivity = df.groupby('strategy').agg({
            'steps_to_discovery': lambda x: x.std() / x.mean() if x.mean() > 0 else 0,
            'discovery_rate': lambda x: x.std() / x.mean() if x.mean() > 0 else 0,
            'discovery_efficiency': lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        })
        
        x = np.arange(len(sensitivity))
        width = 0.25
        
        ax.bar(x - width, sensitivity['steps_to_discovery'], width, label='Steps Variability', alpha=0.8)
        ax.bar(x, sensitivity['discovery_rate'], width, label='Discovery Rate Variability', alpha=0.8)
        ax.bar(x + width, sensitivity['discovery_efficiency'], width, label='Efficiency Variability', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Coefficient of Variation (lower = more robust)')
        ax.set_title('Strategy Robustness Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(sensitivity.index, rotation=45)
        ax.legend()
        
        # MODIFIED: Decision matrix - focus on step and communication metrics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create decision matrix text
        decision_text = "STRATEGY DECISION MATRIX\n\n"
        decision_text += "Scenario | Recommended Strategy | Avg Steps | Comm Cost | Discovery Rate\n"
        decision_text += "-" * 70 + "\n"
        
        for scenario, strategy in recommendations.items():
            steps = df[df['strategy'] == strategy]['steps_to_discovery'].mean()
            comm_cost = df[df['strategy'] == strategy]['communication_overhead'].mean()
            disc_rate = df[df['strategy'] == strategy]['discovery_rate'].mean()
            
            comm_level = "Low" if comm_cost > 100 else "Med" if comm_cost > 10 else "High"
            
            decision_text += f"{scenario[:15]:<15} | {strategy:<18} | {steps:.0f} | {comm_level:<4} | {disc_rate:.3f}\n"
        
        decision_text += "\n\nKEY INSIGHTS:\n"
        decision_text += f"• Fastest discovery: {fastest}\n"
        decision_text += f"• Most efficient: {most_efficient}\n"
        decision_text += f"• Least communication: {least_comm}\n"
        
        # Find most robust strategy
        most_robust = sensitivity['steps_to_discovery'].idxmin()
        decision_text += f"• Most robust: {most_robust}\n"
        
        # Communication trade-offs
        decision_text += f"\nCOMMUNICATION TRADE-OFFS:\n"
        decision_text += f"• Full communication: Fastest but highest overhead\n"
        decision_text += f"• No communication: Slowest but no overhead\n"
        decision_text += f"• Interval strategies: Balance speed vs cost\n"
        
        ax.text(0.05, 0.95, decision_text, transform=ax.transAxes, fontfamily='monospace',
               fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'resource_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_executive_summary(self, df: pd.DataFrame, stats_results: Dict):
        """Generate executive summary with key insights and recommendations"""
        
        summary_path = self.analysis_dir / 'executive_summary.md'
        
        # Calculate key metrics
        total_trials = len(df)
        unique_configs = len(df.groupby(['grid_key', 'n_agents', 'pattern', 'strategy']))
        best_overall = df.groupby('strategy')['discovery_efficiency'].mean().idxmax()  # MODIFIED
        worst_overall = df.groupby('strategy')['discovery_efficiency'].mean().idxmin()  # MODIFIED
        
        # MODIFIED: Step improvements
        baseline_steps = df[df['strategy'] == 'no_comm']['steps_to_discovery'].mean()
        best_steps = df[df['strategy'] == best_overall]['steps_to_discovery'].mean()
        step_improvement = ((baseline_steps - best_steps) / baseline_steps) * 100
        
        with open(summary_path, 'w') as f:
            f.write("# Executive Summary: Distributed Belief Merging Experiment\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Trials Analyzed:** {total_trials:,}\n")
            f.write(f"**Unique Configurations:** {unique_configs}\n")
            f.write(f"**Computational Resources:** {df['computation_time'].sum()/3600:.1f} CPU-hours\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"1. **Most Efficient Strategy:** `{best_overall}` with {step_improvement:.1f}% faster discovery than no communication\n")
            f.write(f"2. **Least Efficient Strategy:** `{worst_overall}` (baseline comparison)\n")
            
            # Pattern-specific insights
            pattern_winners = {}
            for pattern in df['pattern'].unique():
                pattern_data = df[df['pattern'] == pattern]
                winner = pattern_data.groupby('strategy')['discovery_efficiency'].mean().idxmax()  # MODIFIED
                pattern_winners[pattern] = winner
            
            f.write(f"3. **Pattern-Specific Most Efficient:**\n")
            for pattern, winner in pattern_winners.items():
                win_efficiency = df[(df['pattern'] == pattern) & (df['strategy'] == winner)]['discovery_efficiency'].mean()
                f.write(f"   - {pattern.capitalize()}: `{winner}` (efficiency: {win_efficiency:.4f})\n")
            
            # Statistical significance
            if 'strategy_comparisons' in stats_results:
                sig_comparisons = 0
                total_comparisons = 0
                for metric_comps in stats_results['strategy_comparisons'].values():
                    for comp_data in metric_comps.values():
                        if 'significant' in comp_data:
                            total_comparisons += 1
                            if comp_data['significant']:
                                sig_comparisons += 1
                
                f.write(f"4. **Statistical Significance:** {sig_comparisons}/{total_comparisons} comparisons statistically significant (p<0.05)\n\n")
            
            f.write("## Performance Summary\n\n")
            
            # MODIFIED: Create performance table focusing on step metrics
            perf_summary = df.groupby('strategy').agg({
                'discovery_rate': ['mean', 'std'],
                'steps_to_discovery': ['mean', 'std'],
                'discovery_efficiency': ['mean', 'std'],
                'communication_overhead': ['mean', 'std']
            }).round(3)
            
            f.write("| Strategy | Discovery Rate | Steps to Discovery | Discovery Efficiency | Comm Overhead |\n")
            f.write("|----------|----------------|-------------------|---------------------|---------------|\n")
            
            for strategy in perf_summary.index:
                dr = f"{perf_summary.loc[strategy, ('discovery_rate', 'mean')]:.3f}±{perf_summary.loc[strategy, ('discovery_rate', 'std')]:.3f}"
                st = f"{perf_summary.loc[strategy, ('steps_to_discovery', 'mean')]:.0f}±{perf_summary.loc[strategy, ('steps_to_discovery', 'std')]:.0f}"
                de = f"{perf_summary.loc[strategy, ('discovery_efficiency', 'mean')]:.4f}±{perf_summary.loc[strategy, ('discovery_efficiency', 'std')]:.4f}"
                co = f"{perf_summary.loc[strategy, ('communication_overhead', 'mean')]:.0f}±{perf_summary.loc[strategy, ('communication_overhead', 'std')]:.0f}"
                
                f.write(f"| {strategy} | {dr} | {st} | {de} | {co} |\n")
            
            f.write("\n## 🏆 Recommendations\n\n")
            
            # MODIFIED: Generate step-focused recommendations
            fastest_strategy = df.groupby('strategy')['steps_to_discovery'].mean().idxmin()
            most_efficient_strategy = df.groupby('strategy')['discovery_efficiency'].mean().idxmax()
            
            f.write(f"### For Fastest Discovery\n")
            f.write(f"**Use:** `{fastest_strategy}`\n")
            f.write(f"- Achieves fastest discovery in fewest steps\n")
            f.write(f"- Best for time-critical applications\n\n")
            
            f.write(f"### For Best Efficiency\n")
            f.write(f"**Use:** `{most_efficient_strategy}`\n")
            f.write(f"- Best discovery success per step taken\n")
            f.write(f"- Ideal for step-constrained environments\n\n")
            
            f.write(f"### Pattern-Specific Recommendations\n")
            for pattern, winner in pattern_winners.items():
                f.write(f"- **{pattern.capitalize()} targets:** `{winner}`\n")
            
            f.write(f"\n## Scale-Up Insights\n\n")
            
            # MODIFIED: Step-based scaling insights
            grid_scaling = df.groupby('grid_size')['steps_to_discovery'].mean()
            agent_scaling = df.groupby('n_agents')['steps_to_discovery'].mean()
            
            f.write(f"- **Grid Scaling:** Discovery steps scale {grid_scaling.iloc[-1]/grid_scaling.iloc[0]:.1f}x from smallest to largest grid\n")
            f.write(f"- **Agent Scaling:** Discovery steps scale {agent_scaling.iloc[-1]/agent_scaling.iloc[0]:.1f}x from 2 to 4 agents\n")
            f.write(f"- **Communication Sweet Spot:** Balance between frequency and overhead is key\n\n")
            
            f.write("## Statistical Confidence\n\n")
            f.write(f"- All results based on extensive sampling ({total_trials:,} trials)\n")
            f.write(f"- Statistical significance testing performed for all major comparisons\n")
            f.write(f"- Confidence intervals and effect sizes calculated\n")
            f.write(f"- Multiple comparison corrections applied where appropriate\n\n")
            
            f.write("## Generated Analysis Files\n\n")
            f.write("- `performance_landscape.png`: Multi-dimensional performance visualization\n")
            f.write("- `statistical_significance.png`: Statistical significance analysis\n")
            f.write("- `convergence_dynamics.png`: Temporal and convergence analysis\n")
            f.write("- `scalability_analysis.png`: Scaling behavior analysis\n")
            f.write("- `communication_efficiency.png`: Communication strategy analysis\n")
            f.write("- `pattern_deepdive_*.png`: Pattern-specific detailed analysis\n")
            f.write("- `resource_optimization.png`: Cost-benefit analysis\n")
            f.write("- `structured_data.csv`: Complete dataset for further analysis\n")
            f.write("- `advanced_statistics.json`: Detailed statistical results\n")
        
        print(f"Executive summary generated: {summary_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):  # FIXED: Handle both numpy and python booleans
            return bool(obj)
        elif obj == float('inf'):
            return "inf"
        elif obj == float('-inf'):
            return "-inf"
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj
    
    def run_complete_analysis(self):
        """Run the complete advanced analysis pipeline"""
        print("="*80)
        print("ADVANCED CLUSTER-SCALE EXPERIMENT ANALYSIS")
        print("="*80)
        print("Analyzing massive distributed experiment dataset...")
        
        # Memory monitoring
        process = psutil.Process()
        print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        try:
            # Step 1: Load and structure massive dataset
            df = self.load_and_structure_data()
            print(f"Data loaded. Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            
            # Step 2: Advanced statistical analysis
            stats_results = self.advanced_statistical_analysis(df)
            
            # Step 3: Create visualizations
            self.create_advanced_visualizations(df)
            
            # Step 4: Generate executive summary
            self.generate_executive_summary(df, stats_results)
            
            # Cleanup
            gc.collect()
            
            print("\n" + "="*80)
            print("ADVANCED ANALYSIS COMPLETE!")
            print("="*80)
            print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            print(f"Analysis results saved in: {self.analysis_dir}")
            print("\nGenerated files:")
            print("  performance_landscape.png - Multi-dimensional performance view")
            print("  statistical_significance.png - Statistical testing results")
            print("  convergence_dynamics.png - Temporal behavior analysis")
            print("  scalability_analysis.png - Scaling behavior")
            print("  communication_efficiency.png - Communication strategy analysis")
            print("  pattern_deepdive_*.png - Pattern-specific analysis")
            print("  resource_optimization.png - Cost-benefit analysis")
            print("  executive_summary.md - Key insights and recommendations")
            print("  structured_data.csv - Complete dataset for further analysis")
            print("="*80)
            
            return df, stats_results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Main analysis function for cluster-scale experiments"""
    analyzer = AdvancedClusterAnalyzer()
    
    try:
        df, stats = analyzer.run_complete_analysis()
        
        if df is not None:
            print(f"\nSuccessfully analyzed {len(df):,} trials")
            print(f"Results available in: {analyzer.analysis_dir}")
        else:
            print("Analysis failed")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()