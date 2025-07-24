#!/usr/bin/env python3
"""
Analyzer for Multi-Grid Belief Merging Experiments
Handles results from multiple grid sizes and agent numbers
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MultiGridExperimentAnalyzer:
    """Analyzer for experiments with multiple grid sizes and agent numbers"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_consolidated_results(self, filename: str = None) -> Dict:
        """Load consolidated results from pickle file"""
        if filename is None:
            # Find the most recent consolidated results
            result_files = list(self.results_dir.glob("consolidated_results_*.pkl"))
            if not result_files:
                raise FileNotFoundError("No consolidated results found")
            filename = str(max(result_files, key=lambda x: x.stat().st_mtime))
        
        print(f"Loading results from: {filename}")
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def analyze_by_grid_size(self, results: Dict):
        """Analyze performance metrics across different grid sizes"""
        print("\nAnalyzing results by grid size...")
        
        # Prepare data structure
        grid_performance = {}
        
        for grid_key, grid_results in results.items():
            grid_size = grid_key  # e.g., "20x20"
            rows, cols = map(int, grid_key.split('x'))
            total_states = rows * cols
            
            grid_performance[grid_key] = {
                'total_states': total_states,
                'metrics': {}
            }
            
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                
                for pattern, pattern_results in agent_results.items():
                    for strategy, trials in pattern_results.items():
                        if not trials:
                            continue
                        
                        # Calculate average metrics
                        avg_entropy = np.mean([t['final_entropy'] for t in trials])
                        avg_error = np.mean([t['prediction_error'] for t in trials])
                        avg_discovery = np.mean([t['first_discovery_step'] if t['target_found'] else 1000 for t in trials])
                        discovery_rate = np.mean([1 if t['target_found'] else 0 for t in trials])
                        
                        key = f"{n_agents}agents_{pattern}_{strategy}"
                        grid_performance[grid_key]['metrics'][key] = {
                            'avg_entropy': avg_entropy,
                            'avg_error': avg_error,
                            'avg_discovery_step': avg_discovery,
                            'discovery_rate': discovery_rate,
                            'n_trials': len(trials)
                        }
        
        # Create visualizations
        self._plot_grid_size_comparison(grid_performance)
        self._plot_scalability_analysis(grid_performance)
        
        return grid_performance
    
    def analyze_by_agent_number(self, results: Dict):
        """Analyze performance metrics across different agent numbers"""
        print("\nAnalyzing results by agent number...")
        
        agent_performance = {}
        
        for grid_key, grid_results in results.items():
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                
                if n_agents not in agent_performance:
                    agent_performance[n_agents] = {}
                
                for pattern, pattern_results in agent_results.items():
                    for strategy, trials in pattern_results.items():
                        if not trials:
                            continue
                        
                        key = f"{grid_key}_{pattern}_{strategy}"
                        
                        # Calculate metrics
                        metrics = {
                            'avg_entropy': np.mean([t['final_entropy'] for t in trials]),
                            'avg_error': np.mean([t['prediction_error'] for t in trials]),
                            'avg_discovery_step': np.mean([t['first_discovery_step'] if t['target_found'] else 1000 for t in trials]),
                            'discovery_rate': np.mean([1 if t['target_found'] else 0 for t in trials]),
                            'computation_time': np.mean([t['elapsed_time'] for t in trials]),
                            'n_trials': len(trials)
                        }
                        
                        agent_performance[n_agents][key] = metrics
        
        # Create visualizations
        self._plot_agent_number_comparison(agent_performance)
        
        return agent_performance
    
    def analyze_communication_strategies(self, results: Dict):
        """Compare communication strategies across all configurations"""
        print("\nAnalyzing communication strategies...")
        
        # Prepare data for analysis
        strategy_data = []
        
        for grid_key, grid_results in results.items():
            rows, cols = map(int, grid_key.split('x'))
            grid_size = rows * cols
            
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                
                for pattern, pattern_results in agent_results.items():
                    for strategy, trials in pattern_results.items():
                        if not trials:
                            continue
                        
                        for trial in trials:
                            strategy_data.append({
                                'grid_size': grid_size,
                                'grid_key': grid_key,
                                'n_agents': n_agents,
                                'pattern': pattern,
                                'strategy': strategy,
                                'final_entropy': trial['final_entropy'],
                                'prediction_error': trial['prediction_error'],
                                'discovery_step': trial['first_discovery_step'] if trial['target_found'] else 1000,
                                'found': trial['target_found'],
                                'computation_time': trial['elapsed_time']
                            })
        
        df = pd.DataFrame(strategy_data)
        
        # Create comprehensive strategy comparison
        self._plot_strategy_comparison(df)
        
        return df
    
    def _plot_grid_size_comparison(self, grid_performance: Dict):
        """Create plots comparing performance across grid sizes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics vs Grid Size', fontsize=16)
        
        # Extract data for plotting
        grid_sizes = sorted(grid_performance.keys(), 
                           key=lambda x: int(x.split('x')[0]))
        
        metrics_by_strategy = {}
        
        for grid in grid_sizes:
            grid_data = grid_performance[grid]
            for key, metrics in grid_data['metrics'].items():
                if key not in metrics_by_strategy:
                    metrics_by_strategy[key] = {
                        'grid_sizes': [],
                        'total_states': [],
                        'avg_entropy': [],
                        'avg_error': [],
                        'discovery_rate': [],
                        'avg_discovery_step': []
                    }
                
                metrics_by_strategy[key]['grid_sizes'].append(grid)
                metrics_by_strategy[key]['total_states'].append(grid_data['total_states'])
                metrics_by_strategy[key]['avg_entropy'].append(metrics['avg_entropy'])
                metrics_by_strategy[key]['avg_error'].append(metrics['avg_error'])
                metrics_by_strategy[key]['discovery_rate'].append(metrics['discovery_rate'])
                metrics_by_strategy[key]['avg_discovery_step'].append(metrics['avg_discovery_step'])
        
        # Plot 1: Final Entropy vs Grid Size
        ax = axes[0, 0]
        for key, data in metrics_by_strategy.items():
            if len(data['total_states']) > 1:  # Only plot if we have multiple grid sizes
                ax.plot(data['total_states'], data['avg_entropy'], 
                       marker='o', label=key[:20])  # Truncate label
        ax.set_xlabel('Total States in Grid')
        ax.set_ylabel('Average Final Entropy')
        ax.set_title('Final Entropy vs Grid Size')
        ax.set_xscale('log')
        
        # Plot 2: Prediction Error vs Grid Size
        ax = axes[0, 1]
        for key, data in metrics_by_strategy.items():
            if len(data['total_states']) > 1:
                ax.plot(data['total_states'], data['avg_error'], 
                       marker='s', label=key[:20])
        ax.set_xlabel('Total States in Grid')
        ax.set_ylabel('Average Prediction Error')
        ax.set_title('Prediction Error vs Grid Size')
        ax.set_xscale('log')
        
        # Plot 3: Discovery Rate vs Grid Size
        ax = axes[1, 0]
        for key, data in metrics_by_strategy.items():
            if len(data['total_states']) > 1:
                ax.plot(data['total_states'], data['discovery_rate'], 
                       marker='^', label=key[:20])
        ax.set_xlabel('Total States in Grid')
        ax.set_ylabel('Discovery Rate')
        ax.set_title('Discovery Rate vs Grid Size')
        ax.set_xscale('log')
        
        # Plot 4: Discovery Time vs Grid Size
        ax = axes[1, 1]
        for key, data in metrics_by_strategy.items():
            if len(data['total_states']) > 1:
                ax.plot(data['total_states'], data['avg_discovery_step'], 
                       marker='d', label=key[:20])
        ax.set_xlabel('Total States in Grid')
        ax.set_ylabel('Average Discovery Step')
        ax.set_title('Discovery Time vs Grid Size')
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'grid_size_comparison.png', dpi=300)
        plt.close()
    
    def _plot_scalability_analysis(self, grid_performance: Dict):
        """Analyze computational scalability"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis', fontsize=16)
        
        # Prepare scalability data
        grid_sizes = []
        state_counts = []
        
        for grid_key in sorted(grid_performance.keys()):
            rows, cols = map(int, grid_key.split('x'))
            grid_sizes.append(grid_key)
            state_counts.append(rows * cols)
        
        # Plot 1: State Space Growth
        ax1.bar(grid_sizes, state_counts, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Total Number of States')
        ax1.set_title('State Space Growth')
        ax1.set_yscale('log')
        
        # Plot 2: Theoretical Complexity
        # For each grid size and agent number, calculate joint action space
        complexity_data = []
        
        for i, (grid, states) in enumerate(zip(grid_sizes, state_counts)):
            for n_agents in [2, 3, 4]:
                # Approximate number of actions per agent (5: stay + 4 directions)
                actions_per_agent = 5
                joint_actions = actions_per_agent ** n_agents
                complexity = states * joint_actions
                
                complexity_data.append({
                    'grid': grid,
                    'agents': n_agents,
                    'complexity': complexity,
                    'x_pos': i + (n_agents - 3) * 0.25
                })
        
        # Plot complexity bars
        colors = {2: 'lightcoral', 3: 'lightgreen', 4: 'lightskyblue'}
        for n_agents in [2, 3, 4]:
            agent_data = [d for d in complexity_data if d['agents'] == n_agents]
            x_positions = [d['x_pos'] for d in agent_data]
            complexities = [d['complexity'] for d in agent_data]
            ax2.bar(x_positions, complexities, width=0.25, 
                   label=f'{n_agents} agents', color=colors[n_agents])
        
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('Computational Complexity (States × Joint Actions)')
        ax2.set_title('Computational Complexity by Grid Size and Agent Number')
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(grid_sizes)))
        ax2.set_xticklabels(grid_sizes)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'scalability_analysis.png', dpi=300)
        plt.close()
    
    def _plot_agent_number_comparison(self, agent_performance: Dict):
        """Create plots comparing performance across agent numbers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics vs Number of Agents', fontsize=16)
        
        # Prepare data for plotting
        agent_numbers = sorted(agent_performance.keys())
        
        # Aggregate metrics by agent number
        metrics_by_agents = {n: {'entropy': [], 'error': [], 'discovery': [], 'time': []} 
                            for n in agent_numbers}
        
        for n_agents, configs in agent_performance.items():
            for config_key, metrics in configs.items():
                metrics_by_agents[n_agents]['entropy'].append(metrics['avg_entropy'])
                metrics_by_agents[n_agents]['error'].append(metrics['avg_error'])
                metrics_by_agents[n_agents]['discovery'].append(metrics['avg_discovery_step'])
                metrics_by_agents[n_agents]['time'].append(metrics['computation_time'])
        
        # Plot 1: Entropy Distribution
        ax = axes[0, 0]
        data_to_plot = [metrics_by_agents[n]['entropy'] for n in agent_numbers]
        bp = ax.boxplot(data_to_plot, labels=agent_numbers, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Final Entropy')
        ax.set_title('Final Entropy Distribution by Agent Number')
        
        # Plot 2: Prediction Error Distribution
        ax = axes[0, 1]
        data_to_plot = [metrics_by_agents[n]['error'] for n in agent_numbers]
        bp = ax.boxplot(data_to_plot, labels=agent_numbers, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Prediction Error Distribution by Agent Number')
        
        # Plot 3: Discovery Time Distribution
        ax = axes[1, 0]
        data_to_plot = [metrics_by_agents[n]['discovery'] for n in agent_numbers]
        bp = ax.boxplot(data_to_plot, labels=agent_numbers, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Discovery Step')
        ax.set_title('Discovery Time Distribution by Agent Number')
        
        # Plot 4: Computation Time
        ax = axes[1, 1]
        avg_times = [np.mean(metrics_by_agents[n]['time']) for n in agent_numbers]
        std_times = [np.std(metrics_by_agents[n]['time']) for n in agent_numbers]
        ax.bar(agent_numbers, avg_times, yerr=std_times, capsize=10, 
               color='lightyellow', edgecolor='orange')
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Average Computation Time (s)')
        ax.set_title('Computation Time by Agent Number')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'agent_number_comparison.png', dpi=300)
        plt.close()
    
    def _plot_strategy_comparison(self, df: pd.DataFrame):
        """Create comprehensive strategy comparison plots"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Strategy Performance Heatmap
        ax = fig.add_subplot(gs[0, :2])
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(['strategy', 'grid_key'])['final_entropy'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='strategy', columns='grid_key', values='final_entropy')
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis_r', ax=ax)
        ax.set_title('Average Final Entropy by Strategy and Grid Size')
        
        # Plot 2: Agent Number Impact
        ax = fig.add_subplot(gs[0, 2])
        strategy_order = ['full_comm', 'interval_10', 'interval_25', 'interval_50', 
                         'interval_100', 'interval_200', 'no_comm']
        available_strategies = [s for s in strategy_order if s in df['strategy'].unique()]
        
        sns.boxplot(data=df, x='n_agents', y='final_entropy', hue='strategy',
                   hue_order=available_strategies, ax=ax)
        ax.set_title('Entropy by Agent Number and Strategy')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Pattern Performance
        ax = fig.add_subplot(gs[1, 0])
        sns.violinplot(data=df, x='pattern', y='prediction_error', ax=ax)
        ax.set_title('Prediction Error by Movement Pattern')
        
        # Plot 4: Discovery Success Rate
        ax = fig.add_subplot(gs[1, 1])
        discovery_rates = df.groupby(['strategy', 'grid_key'])['found'].mean().reset_index()
        pivot_discovery = discovery_rates.pivot(index='strategy', columns='grid_key', values='found')
        
        sns.heatmap(pivot_discovery, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, 
                   vmin=0, vmax=1, cbar_kws={'label': 'Discovery Rate'})
        ax.set_title('Discovery Success Rate by Strategy and Grid')
        
        # Plot 5: Computation Time Scaling
        ax = fig.add_subplot(gs[1, 2])
        time_data = df.groupby(['grid_key', 'n_agents'])['computation_time'].mean().reset_index()
        
        for n_agents in sorted(df['n_agents'].unique()):
            agent_data = time_data[time_data['n_agents'] == n_agents]
            grid_sizes = [int(g.split('x')[0])**2 for g in agent_data['grid_key']]
            ax.plot(grid_sizes, agent_data['computation_time'], 
                   marker='o', label=f'{n_agents} agents')
        
        ax.set_xlabel('Grid Size (total states)')
        ax.set_ylabel('Average Computation Time (s)')
        ax.set_title('Computation Time Scaling')
        ax.set_xscale('log')
        ax.legend()
        
        # Plot 6: Strategy Efficiency Score
        ax = fig.add_subplot(gs[2, :])
        
        # Calculate efficiency score (lower entropy + higher discovery rate + lower error)
        df['efficiency_score'] = (1 - df['final_entropy']) * df['found'] / (1 + df['prediction_error'])
        
        efficiency_summary = df.groupby(['strategy', 'grid_key', 'n_agents'])['efficiency_score'].mean().reset_index()
        
        # Create grouped bar plot
        grid_agent_combos = [(g, a) for g in sorted(df['grid_key'].unique()) 
                            for a in sorted(df['n_agents'].unique())]
        x_labels = [f"{g}\n{a} agents" for g, a in grid_agent_combos]
        x_positions = np.arange(len(x_labels))
        
        width = 0.8 / len(available_strategies)
        
        for i, strategy in enumerate(available_strategies):
            strategy_data = []
            for grid, agents in grid_agent_combos:
                value = efficiency_summary[
                    (efficiency_summary['strategy'] == strategy) & 
                    (efficiency_summary['grid_key'] == grid) & 
                    (efficiency_summary['n_agents'] == agents)
                ]['efficiency_score'].values
                strategy_data.append(value[0] if len(value) > 0 else 0)
            
            offset = (i - len(available_strategies)/2) * width
            ax.bar(x_positions + offset, strategy_data, width, label=strategy)
        
        ax.set_xlabel('Grid Size and Agent Configuration')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Overall Efficiency Score by Configuration')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'strategy_comparison_comprehensive.png', dpi=300)
        plt.close()
    
    def generate_summary_report(self, results: Dict):
        """Generate a comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MULTI-GRID BELIEF MERGING EXPERIMENT SUMMARY")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Configuration Summary
        report_lines.append("EXPERIMENT CONFIGURATION:")
        report_lines.append("-" * 40)
        
        grid_sizes = list(results.keys())
        agent_numbers = set()
        patterns = set()
        strategies = set()
        
        for grid_key, grid_results in results.items():
            for agent_key, agent_results in grid_results.items():
                agent_numbers.add(int(agent_key.split('_')[0]))
                for pattern, pattern_results in agent_results.items():
                    patterns.add(pattern)
                    strategies.update(pattern_results.keys())
        
        report_lines.append(f"Grid Sizes: {', '.join(sorted(grid_sizes))}")
        report_lines.append(f"Agent Numbers: {', '.join(map(str, sorted(agent_numbers)))}")
        report_lines.append(f"Movement Patterns: {', '.join(sorted(patterns))}")
        report_lines.append(f"Communication Strategies: {', '.join(sorted(strategies))}")
        report_lines.append("")
        
        # Best Configurations
        report_lines.append("BEST PERFORMING CONFIGURATIONS:")
        report_lines.append("-" * 40)
        
        best_configs = self._find_best_configurations(results)
        for metric, config in best_configs.items():
            report_lines.append(f"{metric}: {config}")
        
        report_lines.append("")
        
        # Key Findings
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 40)
        
        findings = self._analyze_key_findings(results)
        for finding in findings:
            report_lines.append(f"• {finding}")
        
        # Save report
        report_path = self.analysis_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to: {report_path}")
        
        # Also print to console
        print("\n".join(report_lines))
    
    def _find_best_configurations(self, results: Dict) -> Dict[str, str]:
        """Find best performing configurations for different metrics"""
        best_configs = {
            'Lowest Entropy': {'value': float('inf'), 'config': ''},
            'Highest Discovery Rate': {'value': 0, 'config': ''},
            'Lowest Prediction Error': {'value': float('inf'), 'config': ''},
            'Fastest Discovery': {'value': float('inf'), 'config': ''}
        }
        
        for grid_key, grid_results in results.items():
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                
                for pattern, pattern_results in agent_results.items():
                    for strategy, trials in pattern_results.items():
                        if not trials:
                            continue
                        
                        config_str = f"{grid_key}, {n_agents} agents, {pattern}, {strategy}"
                        
                        # Calculate metrics
                        avg_entropy = np.mean([t['final_entropy'] for t in trials])
                        discovery_rate = np.mean([1 if t['target_found'] else 0 for t in trials])
                        avg_error = np.mean([t['prediction_error'] for t in trials])
                        avg_discovery = np.mean([t['first_discovery_step'] if t['target_found'] else 1000 for t in trials])
                        
                        # Update best configurations
                        if avg_entropy < best_configs['Lowest Entropy']['value']:
                            best_configs['Lowest Entropy'] = {'value': avg_entropy, 'config': config_str}
                        
                        if discovery_rate > best_configs['Highest Discovery Rate']['value']:
                            best_configs['Highest Discovery Rate'] = {'value': discovery_rate, 'config': config_str}
                        
                        if avg_error < best_configs['Lowest Prediction Error']['value']:
                            best_configs['Lowest Prediction Error'] = {'value': avg_error, 'config': config_str}
                        
                        if avg_discovery < best_configs['Fastest Discovery']['value']:
                            best_configs['Fastest Discovery'] = {'value': avg_discovery, 'config': config_str}
        
        return {k: f"{v['config']} (value: {v['value']:.4f})" for k, v in best_configs.items()}
    
    def _analyze_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from the results"""
        findings = []
        
        # Analyze grid size impact
        grid_sizes = sorted(results.keys(), key=lambda x: int(x.split('x')[0]))
        if len(grid_sizes) > 1:
            findings.append(f"Grid size scaling from {grid_sizes[0]} to {grid_sizes[-1]} shows exponential increase in state space complexity")
        
        # Analyze agent number impact
        agent_performance = {}
        for grid_results in results.values():
            for agent_key, agent_results in grid_results.items():
                n_agents = int(agent_key.split('_')[0])
                if n_agents not in agent_performance:
                    agent_performance[n_agents] = []
                
                for pattern_results in agent_results.values():
                    for trials in pattern_results.values():
                        if trials:
                            agent_performance[n_agents].extend([t['final_entropy'] for t in trials])
        
        if len(agent_performance) > 1:
            agent_nums = sorted(agent_performance.keys())
            entropy_reduction = (np.mean(agent_performance[agent_nums[0]]) - 
                               np.mean(agent_performance[agent_nums[-1]])) / np.mean(agent_performance[agent_nums[0]])
            findings.append(f"Increasing agents from {agent_nums[0]} to {agent_nums[-1]} reduces average entropy by {entropy_reduction*100:.1f}%")
        
        # Communication strategy insights
        strategy_performance = {}
        for grid_results in results.values():
            for agent_results in grid_results.values():
                for pattern_results in agent_results.values():
                    for strategy, trials in pattern_results.items():
                        if trials:
                            if strategy not in strategy_performance:
                                strategy_performance[strategy] = []
                            strategy_performance[strategy].extend([t['final_entropy'] for t in trials])
        
        if 'full_comm' in strategy_performance and 'no_comm' in strategy_performance:
            improvement = (np.mean(strategy_performance['no_comm']) - 
                          np.mean(strategy_performance['full_comm'])) / np.mean(strategy_performance['no_comm'])
            findings.append(f"Full communication reduces entropy by {improvement*100:.1f}% compared to no communication")
        
        # Pattern difficulty
        pattern_difficulty = {}
        for grid_results in results.values():
            for agent_results in grid_results.values():
                for pattern, pattern_results in agent_results.items():
                    if pattern not in pattern_difficulty:
                        pattern_difficulty[pattern] = []
                    
                    for trials in pattern_results.values():
                        if trials:
                            pattern_difficulty[pattern].extend([t['prediction_error'] for t in trials])
        
        if pattern_difficulty:
            easiest = min(pattern_difficulty.items(), key=lambda x: np.mean(x[1]))[0]
            hardest = max(pattern_difficulty.items(), key=lambda x: np.mean(x[1]))[0]
            findings.append(f"Movement patterns ranked by difficulty: {easiest} (easiest) to {hardest} (hardest)")
        
        return findings


def main():
    """Main analysis function"""
    analyzer = MultiGridExperimentAnalyzer()
    
    try:
        # Load results
        results = analyzer.load_consolidated_results()
        
        # Run all analyses
        grid_performance = analyzer.analyze_by_grid_size(results)
        agent_performance = analyzer.analyze_by_agent_number(results)
        strategy_df = analyzer.analyze_communication_strategies(results)
        
        # Generate summary report
        analyzer.generate_summary_report(results)
        
        print("\n" + "="*60)
        print("Analysis complete! Check the 'results/analysis/' directory for:")
        print("  • grid_size_comparison.png")
        print("  • scalability_analysis.png")
        print("  • agent_number_comparison.png")
        print("  • strategy_comparison_comprehensive.png")
        print("  • summary_report_*.txt")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()