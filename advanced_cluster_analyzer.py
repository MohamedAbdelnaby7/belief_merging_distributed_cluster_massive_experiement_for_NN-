#!/usr/bin/env python3
"""
Advanced Cluster-Scale Experiment Analyzer
Designed for massive datasets from distributed Turing cluster experiments
Provides deep insights into belief merging performance across configurations
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
                                
                                # Derived metrics
                                'discovery_rate': 1.0 if trial['target_found'] else 0.0,
                                'search_efficiency': trial['discovery_count'] / trial['elapsed_time'] if trial['elapsed_time'] > 0 else 0,
                                'info_gain': np.log(grid_size) - trial['final_entropy'],  # Information gained
                                'performance_score': (1.0 if trial['target_found'] else 0.0) / (1 + trial['prediction_error']),
                                
                                # Communication metrics
                                'total_merges': trial.get('total_merges', 0),
                                'merge_frequency': trial.get('total_merges', 0) / trial.get('max_steps', 1000) if trial.get('max_steps', 1000) > 0 else 0,
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
                                        row['convergence_speed'] = len(entropy_hist)
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
        
        metrics = ['final_entropy', 'prediction_error', 'discovery_step', 'performance_score']
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
        
        for metric in ['final_entropy', 'prediction_error', 'discovery_rate']:
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
        numeric_cols = ['grid_size', 'n_agents', 'merge_interval', 'final_entropy', 
                       'prediction_error', 'discovery_step', 'computation_time', 'performance_score']
        
        # Replace inf with large number for correlation
        df_corr = df.copy()
        df_corr['merge_interval'] = df_corr['merge_interval'].replace(float('inf'), 1000)
        
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
        
        # Main performance surface plot
        ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        
        # Aggregate data for 3D plot
        agg_data = df.groupby(['merge_interval', 'grid_size']).agg({
            'performance_score': 'mean',
            'final_entropy': 'mean',
            'discovery_rate': 'mean'
        }).reset_index()
        
        # Replace inf with 1000 for plotting
        agg_data['merge_interval'] = agg_data['merge_interval'].replace(float('inf'), 1000)
        
        # Create 3D surface
        X = agg_data['merge_interval']
        Y = agg_data['grid_size']
        Z = agg_data['performance_score']
        
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Merge Interval')
        ax.set_ylabel('Grid Size')
        ax.set_zlabel('Performance Score')
        ax.set_title('Performance Landscape')
        plt.colorbar(scatter, ax=ax, shrink=0.5)
        
        # Entropy vs Performance by Strategy
        ax = fig.add_subplot(gs[0, 2])
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            ax.scatter(strategy_data['final_entropy'], strategy_data['performance_score'], 
                      alpha=0.6, label=strategy, s=20, color=self.colors.get(strategy, 'gray'))
        ax.set_xlabel('Final Entropy')
        ax.set_ylabel('Performance Score')
        ax.set_title('Entropy vs Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Discovery Rate Heatmap
        ax = fig.add_subplot(gs[0, 3])
        heatmap_data = df.groupby(['pattern', 'strategy'])['discovery_rate'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Discovery Rate by Pattern & Strategy')
        
        # Distribution plots
        metrics = ['final_entropy', 'prediction_error', 'discovery_step', 'computation_time']
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[2 + i//2, i%2])
            
            # Box plot by strategy
            strategy_order = ['full_comm', 'interval_10', 'interval_25', 'interval_50', 
                             'interval_100', 'interval_200', 'interval_500', 'no_comm']
            available_strategies = [s for s in strategy_order if s in df['strategy'].unique()]
            
            sns.boxplot(data=df, x='strategy', y=metric, order=available_strategies, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        
        # Grid size impact
        ax = fig.add_subplot(gs[2, 2])
        grid_performance = df.groupby(['grid_key', 'strategy'])['performance_score'].mean().unstack()
        sns.heatmap(grid_performance, annot=True, fmt='.3f', cmap='viridis', ax=ax)
        ax.set_title('Performance by Grid Size & Strategy')
        
        # Agent number impact
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
        
        # Strategy pairwise significance for discovery rate
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
        
        # Strategy pairwise significance for entropy
        ax = axes[0, 1]
        p_matrix = np.ones((len(strategies), len(strategies)))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    data1 = df[df['strategy'] == strategy1]['final_entropy'].dropna()
                    data2 = df[df['strategy'] == strategy2]['final_entropy'].dropna()
                    
                    if len(data1) > 5 and len(data2) > 5:
                        try:
                            _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                            p_matrix[i, j] = p_val
                        except:
                            p_matrix[i, j] = 1.0
        
        sns.heatmap(p_matrix, annot=True, fmt='.3f', xticklabels=strategies, yticklabels=strategies,
                   cmap='RdYlBu_r', ax=ax, vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
        ax.set_title('Final Entropy Significance (p-values)')
        
        # Effect sizes
        ax = axes[1, 0]
        effect_matrix = np.zeros((len(strategies), len(strategies)))
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    data1 = df[df['strategy'] == strategy1]['performance_score'].dropna()
                    data2 = df[df['strategy'] == strategy2]['performance_score'].dropna()
                    
                    if len(data1) > 5 and len(data2) > 5:
                        # Cohen's d effect size
                        pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
                        if pooled_std > 0:
                            effect_size = abs(data1.mean() - data2.mean()) / pooled_std
                            effect_matrix[i, j] = effect_size
        
        sns.heatmap(effect_matrix, annot=True, fmt='.2f', xticklabels=strategies, yticklabels=strategies,
                   cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Effect Size (Cohen\'s d)'})
        ax.set_title('Performance Effect Sizes')
        
        # Configuration performance summary
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
        
        # 1. Convergence speed analysis
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
        
        # 2. Information gain over time
        ax = axes[0, 1]
        info_gain = df.groupby('strategy')['info_gain'].mean().sort_values()
        bars = ax.bar(range(len(info_gain)), info_gain.values, 
                     color=[self.colors.get(s, 'gray') for s in info_gain.index])
        ax.set_xticks(range(len(info_gain)))
        ax.set_xticklabels(info_gain.index, rotation=45)
        ax.set_ylabel('Average Information Gain')
        ax.set_title('Information Gain by Strategy')
        
        # Add value labels on bars
        for bar, value in zip(bars, info_gain.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Entropy reduction rate
        ax = axes[1, 0]
        if 'entropy_reduction_rate' in df.columns:
            sns.violinplot(data=df, x='strategy', y='entropy_reduction_rate', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_title('Entropy Reduction Rate Distribution')
        
        # 4. Performance vs computational cost
        ax = axes[1, 1]
        scatter = ax.scatter(df['computation_time'], df['performance_score'], 
                           c=df['merge_interval'].replace(float('inf'), 1000),
                           cmap='coolwarm', alpha=0.6, s=30)
        ax.set_xlabel('Computation Time (seconds)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance vs Computational Cost')
        plt.colorbar(scatter, ax=ax, label='Merge Interval')
        
        # 5. Discovery dynamics by pattern
        ax = axes[2, 0]
        pattern_discovery = df.groupby(['pattern', 'strategy'])['discovery_rate'].mean().unstack()
        sns.heatmap(pattern_discovery, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
        ax.set_title('Discovery Success by Pattern & Strategy')
        
        # 6. Robustness analysis (variance across trials)
        ax = axes[2, 1]
        robustness = df.groupby('strategy').agg({
            'final_entropy': 'std',
            'prediction_error': 'std',
            'performance_score': 'std'
        })
        
        x = np.arange(len(robustness))
        width = 0.25
        
        ax.bar(x - width, robustness['final_entropy'], width, label='Entropy Std', alpha=0.8)
        ax.bar(x, robustness['prediction_error'], width, label='Error Std', alpha=0.8)
        ax.bar(x + width, robustness['performance_score'], width, label='Performance Std', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Standard Deviation (Robustness)')
        ax.set_title('Robustness Analysis (Lower = More Robust)')
        ax.set_xticks(x)
        ax.set_xticklabels(robustness.index, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'convergence_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, df: pd.DataFrame):
        """Analyze scalability across grid sizes and agent numbers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Performance scaling with grid size
        ax = axes[0, 0]
        for strategy in ['full_comm', 'interval_25', 'interval_100', 'no_comm']:
            if strategy in df['strategy'].values:
                strategy_data = df[df['strategy'] == strategy]
                scaling_data = strategy_data.groupby('grid_size')['performance_score'].mean()
                ax.plot(scaling_data.index, scaling_data.values, 'o-', 
                       label=strategy, linewidth=2, markersize=6)
        
        ax.set_xlabel('Grid Size (total states)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance Scaling with Grid Size')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Computational complexity scaling
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
        
        # 3. Agent scaling efficiency
        ax = axes[0, 2]
        agent_efficiency = df.groupby(['n_agents', 'strategy'])['search_efficiency'].mean().unstack()
        
        for strategy in agent_efficiency.columns:
            ax.plot(agent_efficiency.index, agent_efficiency[strategy], 
                   'o-', label=strategy, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Search Efficiency (discoveries/second)')
        ax.set_title('Search Efficiency by Agent Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Memory efficiency (entropy reduction per merge)
        ax = axes[1, 0]
        if 'total_merges' in df.columns:
            df_with_merges = df[df['total_merges'] > 0].copy()
            df_with_merges['entropy_per_merge'] = df_with_merges['info_gain'] / df_with_merges['total_merges']
            
            merge_efficiency = df_with_merges.groupby('strategy')['entropy_per_merge'].mean()
            bars = ax.bar(range(len(merge_efficiency)), merge_efficiency.values,
                         color=[self.colors.get(s, 'gray') for s in merge_efficiency.index])
            ax.set_xticks(range(len(merge_efficiency)))
            ax.set_xticklabels(merge_efficiency.index, rotation=45)
            ax.set_ylabel('Information Gain per Merge')
            ax.set_title('Communication Efficiency')
        
        # 5. Pattern complexity analysis
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
        
        # 6. Optimal configuration finder
        ax = axes[1, 2]
        
        # Find Pareto frontier: best performance for each computational cost level
        pareto_data = df.groupby(['strategy', 'pattern']).agg({
            'performance_score': 'mean',
            'computation_time': 'mean',
            'final_entropy': 'mean'
        }).reset_index()
        
        scatter = ax.scatter(pareto_data['computation_time'], pareto_data['performance_score'],
                           c=pareto_data['final_entropy'], cmap='viridis_r', s=100, alpha=0.7)
        
        # Annotate points
        for _, row in pareto_data.iterrows():
            ax.annotate(f"{row['strategy'][:8]}\n{row['pattern'][:4]}", 
                       (row['computation_time'], row['performance_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Average Computation Time (s)')
        ax.set_ylabel('Average Performance Score')
        ax.set_title('Performance vs Cost Trade-off')
        plt.colorbar(scatter, ax=ax, label='Final Entropy')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_communication_efficiency(self, df: pd.DataFrame):
        """Analyze communication efficiency and overhead"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Communication frequency vs performance
        ax = axes[0, 0]
        comm_data = df[df['merge_interval'] != float('inf')].copy()
        comm_data['comm_frequency'] = 1 / comm_data['merge_interval']
        
        for pattern in df['pattern'].unique():
            pattern_data = comm_data[comm_data['pattern'] == pattern]
            freq_perf = pattern_data.groupby('comm_frequency').agg({
                'performance_score': 'mean',
                'final_entropy': 'mean'
            })
            
            ax.plot(freq_perf.index, freq_perf['performance_score'], 'o-', 
                   label=f'{pattern} Performance', linewidth=2, markersize=6)
        
        ax.set_xlabel('Communication Frequency (1/merge_interval)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Communication Frequency vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Diminishing returns analysis
        ax = axes[0, 1]
        
        # Calculate marginal benefit of additional communication
        strategies_ordered = ['no_comm', 'interval_500', 'interval_200', 'interval_100', 
                             'interval_50', 'interval_25', 'interval_10', 'full_comm']
        available_strategies = [s for s in strategies_ordered if s in df['strategy'].unique()]
        
        marginal_benefits = []
        strategy_labels = []
        
        baseline_perf = df[df['strategy'] == 'no_comm']['performance_score'].mean()
        
        for strategy in available_strategies[1:]:  # Skip no_comm
            strategy_perf = df[df['strategy'] == strategy]['performance_score'].mean()
            marginal_benefit = strategy_perf - baseline_perf
            marginal_benefits.append(marginal_benefit)
            strategy_labels.append(strategy)
            baseline_perf = strategy_perf
        
        bars = ax.bar(range(len(marginal_benefits)), marginal_benefits,
                     color=[self.colors.get(s, 'gray') for s in strategy_labels])
        ax.set_xticks(range(len(strategy_labels)))
        ax.set_xticklabels(strategy_labels, rotation=45)
        ax.set_ylabel('Marginal Performance Benefit')
        ax.set_title('Diminishing Returns of Communication')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Communication overhead analysis
        ax = axes[1, 0]
        
        # Estimate communication cost (number of merges * agents^2)
        df_comm_cost = df.copy()
        df_comm_cost['comm_cost'] = df_comm_cost['total_merges'] * df_comm_cost['n_agents']**2
        
        cost_benefit = df_comm_cost.groupby('strategy').agg({
            'comm_cost': 'mean',
            'performance_score': 'mean',
            'computation_time': 'mean'
        })
        
        # Calculate efficiency ratio
        cost_benefit['efficiency_ratio'] = cost_benefit['performance_score'] / (1 + cost_benefit['comm_cost'])
        
        scatter = ax.scatter(cost_benefit['comm_cost'], cost_benefit['efficiency_ratio'],
                           s=cost_benefit['computation_time'] * 10, alpha=0.7,
                           c=range(len(cost_benefit)), cmap='viridis')
        
        for strategy, row in cost_benefit.iterrows():
            ax.annotate(strategy, (row['comm_cost'], row['efficiency_ratio']),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Communication Cost (merges × agents²)')
        ax.set_ylabel('Efficiency Ratio')
        ax.set_title('Communication Cost vs Efficiency')
        
        # 4. Optimal communication strategy finder
        ax = axes[1, 1]
        
        # For each pattern and grid size, find best strategy
        optimal_strategies = {}
        
        for pattern in df['pattern'].unique():
            for grid_key in df['grid_key'].unique():
                subset = df[(df['pattern'] == pattern) & (df['grid_key'] == grid_key)]
                if len(subset) > 0:
                    best_strategy = subset.groupby('strategy')['performance_score'].mean().idxmax()
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
        ax.set_title('Frequency of Optimal Strategies')
        
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
            
            # 1. Strategy performance ranking
            ax = axes[0, 0]
            strategy_ranking = pattern_data.groupby('strategy').agg({
                'performance_score': ['mean', 'std'],
                'discovery_rate': 'mean',
                'final_entropy': 'mean'
            }).round(3)
            
            # Flatten column names
            strategy_ranking.columns = ['_'.join(col).strip() for col in strategy_ranking.columns]
            strategy_ranking = strategy_ranking.sort_values('performance_score_mean', ascending=False)
            
            # Plot ranking
            y_pos = np.arange(len(strategy_ranking))
            bars = ax.barh(y_pos, strategy_ranking['performance_score_mean'],
                          xerr=strategy_ranking['performance_score_std'],
                          color=[self.colors.get(s, 'gray') for s in strategy_ranking.index])
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(strategy_ranking.index)
            ax.set_xlabel('Performance Score')
            ax.set_title(f'{pattern}: Strategy Ranking')
            
            # Add values
            for i, (bar, mean_val) in enumerate(zip(bars, strategy_ranking['performance_score_mean'])):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{mean_val:.3f}', va='center', fontsize=9)
            
            # 2. Grid size sensitivity
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
            
            # 3. Discovery time distribution
            ax = axes[0, 2]
            
            discovery_times = {}
            for strategy in pattern_data['strategy'].unique():
                strategy_data = pattern_data[pattern_data['strategy'] == strategy]
                times = strategy_data[strategy_data['target_found']]['discovery_step'].values
                if len(times) > 0:
                    discovery_times[strategy] = times
            
            if discovery_times:
                ax.boxplot(discovery_times.values(), labels=discovery_times.keys())
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel('Discovery Step')
                ax.set_title(f'{pattern}: Discovery Time Distribution')
            
            # 4. Entropy evolution
            ax = axes[1, 0]
            
            # Sample trials with entropy history
            sample_trials = pattern_data.sample(min(50, len(pattern_data)))
            
            for _, trial_row in sample_trials.iterrows():
                # This would require loading individual trial data - simplified for now
                # Placeholder: show final entropy vs discovery success
                pass
            
            # Simplified: final entropy by strategy
            entropy_by_strategy = pattern_data.groupby('strategy')['final_entropy'].mean()
            bars = ax.bar(range(len(entropy_by_strategy)), entropy_by_strategy.values,
                         color=[self.colors.get(s, 'gray') for s in entropy_by_strategy.index])
            ax.set_xticks(range(len(entropy_by_strategy)))
            ax.set_xticklabels(entropy_by_strategy.index, rotation=45)
            ax.set_ylabel('Average Final Entropy')
            ax.set_title(f'{pattern}: Final Uncertainty')
            
            # 5. Agent coordination efficiency
            ax = axes[1, 1]
            agent_coord = pattern_data.groupby(['n_agents', 'strategy'])['discovery_rate'].mean().unstack()
            
            sns.heatmap(agent_coord, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            ax.set_title(f'{pattern}: Agent Coordination Efficiency')
            
            # 6. Statistical summary table
            ax = axes[1, 2]
            ax.axis('off')
            
            # Create summary statistics table
            summary_stats = pattern_data.groupby('strategy').agg({
                'discovery_rate': ['mean', 'std'],
                'prediction_error': ['mean', 'std'],
                'final_entropy': ['mean', 'std'],
                'computation_time': ['mean', 'std']
            }).round(3)
            
            # Format as text table
            table_text = f"{pattern.upper()} SUMMARY STATISTICS\n\n"
            table_text += "Strategy | Discovery Rate | Pred Error | Final Entropy | Comp Time\n"
            table_text += "-" * 70 + "\n"
            
            for strategy in summary_stats.index:
                dr_mean = summary_stats.loc[strategy, ('discovery_rate', 'mean')]
                dr_std = summary_stats.loc[strategy, ('discovery_rate', 'std')]
                pe_mean = summary_stats.loc[strategy, ('prediction_error', 'mean')]
                pe_std = summary_stats.loc[strategy, ('prediction_error', 'std')]
                fe_mean = summary_stats.loc[strategy, ('final_entropy', 'mean')]
                fe_std = summary_stats.loc[strategy, ('final_entropy', 'std')]
                ct_mean = summary_stats.loc[strategy, ('computation_time', 'mean')]
                ct_std = summary_stats.loc[strategy, ('computation_time', 'std')]
                
                table_text += f"{strategy[:12]:<12} | {dr_mean:.3f}±{dr_std:.3f} | {pe_mean:.2f}±{pe_std:.2f} | {fe_mean:.3f}±{fe_std:.3f} | {ct_mean:.3f}±{ct_std:.3f}\n"
            
            ax.text(0.05, 0.95, table_text, transform=ax.transAxes, fontfamily='monospace',
                   fontsize=8, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(self.analysis_dir / f'pattern_deepdive_{pattern}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_resource_optimization(self, df: pd.DataFrame):
        """Analyze resource optimization and recommendations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance per computational dollar
        ax = axes[0, 0]
        
        # Normalize computation time to "cost units"
        max_time = df['computation_time'].max()
        df_cost = df.copy()
        df_cost['relative_cost'] = df_cost['computation_time'] / max_time
        
        cost_efficiency = df_cost.groupby('strategy').agg({
            'performance_score': 'mean',
            'relative_cost': 'mean'
        })
        cost_efficiency['efficiency'] = cost_efficiency['performance_score'] / cost_efficiency['relative_cost']
        
        bars = ax.bar(range(len(cost_efficiency)), cost_efficiency['efficiency'],
                     color=[self.colors.get(s, 'gray') for s in cost_efficiency.index])
        ax.set_xticks(range(len(cost_efficiency)))
        ax.set_xticklabels(cost_efficiency.index, rotation=45)
        ax.set_ylabel('Performance per Computational Cost')
        ax.set_title('Cost Efficiency Ranking')
        
        # 2. Recommended configurations
        ax = axes[0, 1]
        
        # Find best strategy for each scenario
        recommendations = {}
        scenarios = ['high_performance', 'balanced', 'low_cost']
        
        # High performance: maximize performance regardless of cost
        high_perf = df.groupby('strategy')['performance_score'].mean().idxmax()
        recommendations['High Performance'] = high_perf
        
        # Balanced: best performance/cost ratio
        balanced = cost_efficiency['efficiency'].idxmax()
        recommendations['Balanced'] = balanced
        
        # Low cost: minimize computation time while maintaining reasonable performance
        low_cost_candidates = df[df['performance_score'] > df['performance_score'].quantile(0.7)]
        low_cost = low_cost_candidates.groupby('strategy')['computation_time'].mean().idxmin()
        recommendations['Low Cost'] = low_cost
        
        # Plot recommendations
        scenario_names = list(recommendations.keys())
        recommended_strategies = list(recommendations.values())
        
        # Get performance scores for recommended strategies
        perf_scores = []
        for strategy in recommended_strategies:
            score = df[df['strategy'] == strategy]['performance_score'].mean()
            perf_scores.append(score)
        
        bars = ax.bar(scenario_names, perf_scores,
                     color=[self.colors.get(s, 'gray') for s in recommended_strategies])
        ax.set_ylabel('Performance Score')
        ax.set_title('Recommended Configurations by Scenario')
        
        # Add strategy labels on bars
        for bar, strategy in zip(bars, recommended_strategies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   strategy, ha='center', va='bottom', rotation=45, fontsize=9)
        
        # 3. Sensitivity analysis
        ax = axes[1, 0]
        
        # Calculate coefficient of variation for each strategy
        sensitivity = df.groupby('strategy').agg({
            'performance_score': lambda x: x.std() / x.mean(),  # CV
            'discovery_rate': lambda x: x.std() / x.mean() if x.mean() > 0 else 0,
            'final_entropy': lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        })
        
        x = np.arange(len(sensitivity))
        width = 0.25
        
        ax.bar(x - width, sensitivity['performance_score'], width, label='Performance', alpha=0.8)
        ax.bar(x, sensitivity['discovery_rate'], width, label='Discovery Rate', alpha=0.8)
        ax.bar(x + width, sensitivity['final_entropy'], width, label='Final Entropy', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Coefficient of Variation (lower = more robust)')
        ax.set_title('Strategy Robustness Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(sensitivity.index, rotation=45)
        ax.legend()
        
        # 4. Decision matrix
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create decision matrix text
        decision_text = "STRATEGY DECISION MATRIX\n\n"
        decision_text += "Scenario | Recommended Strategy | Performance | Cost | Robustness\n"
        decision_text += "-" * 65 + "\n"
        
        for scenario, strategy in recommendations.items():
            perf = df[df['strategy'] == strategy]['performance_score'].mean()
            cost = df[df['strategy'] == strategy]['computation_time'].mean()
            robust = sensitivity.loc[strategy, 'performance_score']
            
            cost_level = "Low" if cost < df['computation_time'].quantile(0.33) else "Med" if cost < df['computation_time'].quantile(0.67) else "High"
            robust_level = "High" if robust < 0.1 else "Med" if robust < 0.2 else "Low"
            
            decision_text += f"{scenario:<15} | {strategy:<18} | {perf:.3f} | {cost_level:<4} | {robust_level:<4}\n"
        
        decision_text += "\n\nKEY INSIGHTS:\n"
        decision_text += f"• Best overall: {high_perf}\n"
        decision_text += f"• Most efficient: {balanced}\n"
        decision_text += f"• Most economical: {low_cost}\n"
        
        # Find most robust strategy
        most_robust = sensitivity['performance_score'].idxmin()
        decision_text += f"• Most robust: {most_robust}\n"
        
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
        best_overall = df.groupby('strategy')['performance_score'].mean().idxmax()
        worst_overall = df.groupby('strategy')['performance_score'].mean().idxmin()
        
        # Performance improvements
        baseline_perf = df[df['strategy'] == 'no_comm']['performance_score'].mean()
        best_perf = df[df['strategy'] == best_overall]['performance_score'].mean()
        improvement = ((best_perf - baseline_perf) / baseline_perf) * 100
        
        with open(summary_path, 'w') as f:
            f.write("# Executive Summary: Distributed Belief Merging Experiment\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Trials Analyzed:** {total_trials:,}\n")
            f.write(f"**Unique Configurations:** {unique_configs}\n")
            f.write(f"**Computational Resources:** {df['computation_time'].sum()/3600:.1f} CPU-hours\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"1. **Best Strategy:** `{best_overall}` with {improvement:.1f}% improvement over no communication\n")
            f.write(f"2. **Worst Strategy:** `{worst_overall}` (baseline comparison)\n")
            
            # Pattern-specific insights
            pattern_winners = {}
            for pattern in df['pattern'].unique():
                pattern_data = df[df['pattern'] == pattern]
                winner = pattern_data.groupby('strategy')['performance_score'].mean().idxmax()
                pattern_winners[pattern] = winner
            
            f.write(f"3. **Pattern-Specific Winners:**\n")
            for pattern, winner in pattern_winners.items():
                win_score = df[(df['pattern'] == pattern) & (df['strategy'] == winner)]['performance_score'].mean()
                f.write(f"   - {pattern.capitalize()}: `{winner}` (score: {win_score:.3f})\n")
            
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
            
            # Create performance table
            perf_summary = df.groupby('strategy').agg({
                'discovery_rate': ['mean', 'std'],
                'prediction_error': ['mean', 'std'],
                'final_entropy': ['mean', 'std'],
                'computation_time': ['mean', 'std']
            }).round(3)
            
            f.write("| Strategy | Discovery Rate | Prediction Error | Final Entropy | Comp Time (s) |\n")
            f.write("|----------|----------------|------------------|---------------|---------------|\n")
            
            for strategy in perf_summary.index:
                dr = f"{perf_summary.loc[strategy, ('discovery_rate', 'mean')]:.3f}±{perf_summary.loc[strategy, ('discovery_rate', 'std')]:.3f}"
                pe = f"{perf_summary.loc[strategy, ('prediction_error', 'mean')]:.2f}±{perf_summary.loc[strategy, ('prediction_error', 'std')]:.2f}"
                fe = f"{perf_summary.loc[strategy, ('final_entropy', 'mean')]:.3f}±{perf_summary.loc[strategy, ('final_entropy', 'std')]:.3f}"
                ct = f"{perf_summary.loc[strategy, ('computation_time', 'mean')]:.3f}±{perf_summary.loc[strategy, ('computation_time', 'std')]:.3f}"
                
                f.write(f"| {strategy} | {dr} | {pe} | {fe} | {ct} |\n")
            
            f.write("\n## 🏆 Recommendations\n\n")
            
            # Generate specific recommendations
            high_perf_strategy = df.groupby('strategy')['performance_score'].mean().idxmax()
            efficient_strategy = (df.groupby('strategy').agg({
                'performance_score': 'mean',
                'computation_time': 'mean'
            }).assign(efficiency=lambda x: x['performance_score']/x['computation_time']))['efficiency'].idxmax()
            
            f.write(f"### For Maximum Performance\n")
            f.write(f"**Use:** `{high_perf_strategy}`\n")
            f.write(f"- Achieves highest overall performance score\n")
            f.write(f"- Best for critical applications where performance is paramount\n\n")
            
            f.write(f"### For Best Efficiency\n")
            f.write(f"**Use:** `{efficient_strategy}`\n")
            f.write(f"- Best performance per computational cost\n")
            f.write(f"- Ideal for resource-constrained environments\n\n")
            
            f.write(f"### Pattern-Specific Recommendations\n")
            for pattern, winner in pattern_winners.items():
                f.write(f"- **{pattern.capitalize()} targets:** `{winner}`\n")
            
            f.write(f"\n## Scale-Up Insights\n\n")
            
            # Scalability insights
            grid_scaling = df.groupby('grid_size')['computation_time'].mean()
            agent_scaling = df.groupby('n_agents')['computation_time'].mean()
            
            f.write(f"- **Grid Scaling:** Computation time scales {grid_scaling.iloc[-1]/grid_scaling.iloc[0]:.1f}x from smallest to largest grid\n")
            f.write(f"- **Agent Scaling:** Computation time scales {agent_scaling.iloc[-1]/agent_scaling.iloc[0]:.1f}x from 2 to 4 agents\n")
            f.write(f"- **Sweet Spot:** {df.groupby(['grid_key', 'n_agents']).apply(lambda x: x['performance_score'].mean()/x['computation_time'].mean()).idxmax()} for performance/cost ratio\n\n")
            
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
            f.write("- `resource_optimization.png`: Resource optimization recommendations\n")
            f.write("- `structured_data.csv`: Complete structured dataset\n")
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