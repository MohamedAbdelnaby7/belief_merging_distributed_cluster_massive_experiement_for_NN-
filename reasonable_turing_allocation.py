#!/usr/bin/env python3
"""
Reasonable Resource Allocation for Shared University Cluster
Modified to support multiple grid sizes and agent numbers
Designed to get scheduled quickly while still providing significant speedup
"""

import json
import os
from pathlib import Path


class ReasonableResourceManager:
    """Conservative resource management for shared university clusters with multiple configurations"""
    
    def __init__(self):
        # Conservative estimates for shared cluster usage
        self.reasonable_limits = {
            'max_cores_per_job': 128,        # Reasonable for university cluster
            'max_memory_gb': 256,            # Conservative memory request
            'max_time_hours': 150,            # max to get scheduled
            'cores_per_node': 32,            # Typical node size
        }
    
    def calculate_reasonable_resources(self, experiment_type: str = "standard") -> dict:
        """Calculate reasonable resource allocation that will actually get scheduled"""
        
        # Different tiers of resource usage
        resource_tiers = {
            # Quick testing - gets scheduled immediately
            'test': {
                'cores': 8,
                'memory_gb': 16,
                'time_hours': 2,
                'nodes': 1,
                'partition': 'short',
                'description': 'Quick testing and debugging'
            },
            
            # Small scale - scheduled within hours
            'small': {
                'cores': 32,
                'memory_gb': 64,
                'time_hours': 12,
                'nodes': 1,
                'partition': 'short',
                'description': 'Small scale experiment'
            },
            
            # Standard - good balance, scheduled within a day
            'standard': {
                'cores': 64,
                'memory_gb': 128,
                'time_hours': 24,
                'nodes': 2,
                'partition': 'long',
                'description': 'Standard full experiment'
            },
            
            # Large - comprehensive, may wait 1-2 days for scheduling
            'large': {
                'cores': 180,
                'memory_gb': 256,
                'time_hours': 120,
                'nodes': 4,
                'partition': 'long',
                'description': 'Large comprehensive study'
            }
        }
        
        return resource_tiers.get(experiment_type, resource_tiers['standard'])
    
    def create_reasonable_config(self, experiment_type: str = "standard") -> dict:
        """Create experiment configuration with proper MPC (no fast mode) for multiple grid sizes"""
        
        # Configuration based on available compute time
        config_templates = {
            'test': {
                'grid_sizes': [[10, 10], [20, 20]],  # Small grids for testing
                'n_agents_list': [2, 3],  # Fewer agents for quick test
                'n_trials': 5,
                'horizon': 2,
                'max_steps': 200,
                'merge_intervals': [0, 50, float('inf')],
                'target_patterns': ['random'],
                'fast_mode': False,  # TRUE MPC
                'random_walk_mode': False, # Active search
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            },
            
            'small': {
                'grid_sizes': [[10, 10], [20, 20]],  # Small to medium grids
                'n_agents_list': [2, 3, 4],  # All agent numbers
                'n_trials': 20,
                'horizon': 2,
                'max_steps': 500,
                'merge_intervals': [0, 25, 100, float('inf')],
                'target_patterns': ['random', 'evasive'],
                'fast_mode': False,  # TRUE MPC
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            },
            
            'standard': {
                'grid_sizes': [[10, 10], [20, 20], [30, 30]],  # Full range of grids
                'n_agents_list': [2, 3, 4],  # All agent numbers
                'n_trials': 50,
                'horizon': 3,  # Full MPC horizon
                'max_steps': 1000,
                'merge_intervals': [0, 10, 25, 50, 100, 200, float('inf')],
                'target_patterns': ['random', 'evasive', 'patrol'],
                'fast_mode': False,  # TRUE MPC - computationally intensive
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            },
            
            'large': {
                'grid_sizes': [[10, 10], [15, 15], [20, 20], [25, 25], [30, 30], [40, 40], [45, 45], [50, 50], [100, 100]],  # Extended grid sizes
                'n_agents_list': [2, 3, 4],  # All agent numbers
                'n_trials': 10,
                'horizon': 3,  # Full MPC horizon
                'max_steps': 1000,
                'merge_intervals': [0, 5, 10, 25, 50, 100, 200, 500, float('inf')],
                'target_patterns': ['random', 'evasive', 'patrol'],
                'fast_mode': False,  # TRUE MPC
                'random_walk_mode': False,
                'merge_methods': ['standard_kl', 'reverse_kl', 'geometric_mean', 'arithmetic_mean']
            }
        }
        
        base_config = config_templates.get(experiment_type, config_templates['standard'])
        
        # Add common parameters
        full_config = {
            'alpha': 0.1,  # False positive rate
            'beta': 0.2,   # False negative rate
            **base_config
        }
        
        return full_config
    
    def create_slurm_script(self, experiment_type: str = "standard") -> str:
        """Create reasonable SLURM script that will get scheduled"""
        
        resources = self.calculate_reasonable_resources(experiment_type)
        
        script_content = f"""#!/bin/bash
#SBATCH -N {resources['nodes']}
#SBATCH -n {resources['cores']}
#SBATCH --mem={resources['memory_gb']}G
#SBATCH -J "BeliefMerging-{experiment_type}"
#SBATCH -p {resources['partition']}
#SBATCH -t {resources['time_hours']:02d}:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@wpi.edu

# Conservative resource allocation for shared cluster
echo "=========================================="
echo "Belief Merging Experiment - {experiment_type.upper()}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: {resources['nodes']}"
echo "CPU Cores: {resources['cores']}"
echo "Memory: {resources['memory_gb']}G"
echo "Time Limit: {resources['time_hours']} hours"
echo "Partition: {resources['partition']}"
echo "Description: {resources['description']}"
echo "=========================================="

# Load modules efficiently
module load python/3.8 2>/dev/null || module load Python/3.8 2>/dev/null || echo "Using system Python"

# Environment setup
cd $SLURM_SUBMIT_DIR

# Load Python 3.10 module (where packages are installed)
module load python/3.10.17/v6xrl7k 2>/dev/null || echo "Could not load Python 3.10 module"

# Setup Python environment
export PATH=/home/mabdelnaby/.local/bin:$PATH
export PYTHONPATH=/home/mabdelnaby/.local/lib/python3.10/site-packages:$PYTHONPATH

# Set environment for optimal performance with limited resources
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Pre-flight checks
echo "Pre-flight checks:"
echo "  Python: $(python3 --version)"
echo "  Working directory: $(pwd)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{{print $7}}')"
echo "  Disk space: $(df -h . | tail -1 | awk '{{print $4}}')"

# Clean up any previous partial runs
find checkpoints/ -name "*.tmp" -delete 2>/dev/null || true
find checkpoints/ -name "*.pkl" -size 0 -delete 2>/dev/null || true

# Start monitoring (lightweight)
(
    while true; do
        echo "$(date): CPU=$(top -bn1 | grep "Cpu(s)" | awk '{{print $2}}' | cut -d'%' -f1)% MEM=$(free | grep Mem | awk '{{printf "%.1f%%", $3/$2*100}}')" >> logs/resource_usage.log
        sleep 300  # Every 5 minutes
    done
) &
MONITOR_PID=$!

# Run experiment with TRUE MPC (no fast mode)
echo "Starting TRUE MPC experiment at $(date)"
echo "WARNING: True MPC is computationally intensive - will take longer but give accurate results"

python3 complete_distributed_experiment.py \\
    --config-file configs/{experiment_type}_config.json \\
    --max-workers {resources['cores']} \\
    --checkpoint-dir checkpoints \\
    --results-dir results

EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# Final summary
echo "=========================================="
echo "Experiment completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Count results
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

echo "Results Summary:"
echo "  Completed tasks: $COMPLETED"
echo "  Failed tasks: $ERRORS"
echo "  Log files in: logs/"
echo "  Results in: results/"

# Generate final analysis if enough tasks completed
if [[ $COMPLETED -gt 10 ]]; then
    echo "Generating analysis..."
    python3 -c "
from complete_distributed_experiment import DistributedExperimentManager, ExperimentConfig
import json

# Load config and analyze
with open('configs/{experiment_type}_config.json', 'r') as f:
    config_dict = json.load(f)

config = ExperimentConfig.from_dict(config_dict)
manager = DistributedExperimentManager(config, 'checkpoints', 'results')
results = manager.collect_results()

if results:
    print('Analysis generated successfully')
else:
    print('Analysis generation failed')
"
fi

echo "Job completed!"
exit $EXIT_CODE
"""
        
        return script_content
    
    def estimate_runtime(self, experiment_type: str = "standard") -> dict:
        """Estimate realistic runtime with true MPC for multiple configurations"""
        
        config = self.create_reasonable_config(experiment_type)
        resources = self.calculate_reasonable_resources(experiment_type)
        
        # Calculate total tasks
        n_grid_sizes = len(config['grid_sizes'])
        n_agent_configs = len(config['n_agents_list'])
        n_intervals = len(config['merge_intervals'])
        n_patterns = len(config['target_patterns'])
        n_methods = len(config.get('merge_methods', ['standard_kl']))
        n_trials = config['n_trials']
        total_tasks = n_grid_sizes * n_agent_configs * n_intervals * n_patterns * n_trials * n_methods
        
        # TRUE MPC is much slower than fast mode
        # Time scales with grid size and number of agents
        avg_grid_size = sum(r*c for r,c in config['grid_sizes']) / n_grid_sizes
        avg_agents = sum(config['n_agents_list']) / len(config['n_agents_list'])
        
        # Base time per task depends on grid size and agents
        if config['horizon'] <= 2:
            base_time = 10  # seconds for small problems
        else:
            base_time = 20  # seconds for horizon 3
        
        # Scale by grid size and agent number
        time_per_task = base_time * (avg_grid_size / 400) * (avg_agents / 3)
        
        # Total sequential time
        total_time_sequential = total_tasks * time_per_task
        
        # Parallel time
        total_time_parallel = total_time_sequential / resources['cores']
        
        # Add overhead (checkpointing, I/O, etc.)
        overhead_factor = 1.3
        estimated_time = total_time_parallel * overhead_factor
        
        return {
            'total_tasks': total_tasks,
            'time_per_task_seconds': time_per_task,
            'sequential_hours': total_time_sequential / 3600,
            'parallel_hours': estimated_time / 3600,
            'speedup': total_time_sequential / estimated_time,
            'cores_used': resources['cores'],
            'memory_used_gb': resources['memory_gb'],
            'will_fit_in_time_limit': estimated_time / 3600 < resources['time_hours'],
            'n_grid_sizes': n_grid_sizes,
            'n_agent_configs': n_agent_configs
        }
    
    def generate_all_configurations(self):
        """Generate all reasonable configurations"""
        
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)
        
        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        experiment_types = ['test', 'small', 'standard', 'large']
        
        print("Generating reasonable configurations for shared cluster:")
        print("="*60)
        
        for exp_type in experiment_types:
            # Generate config
            config = self.create_reasonable_config(exp_type)
            config_path = configs_dir / f"{exp_type}_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Generate SLURM script
            script_content = self.create_slurm_script(exp_type)
            script_path = scripts_dir / f"run_{exp_type}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            # Print estimates
            estimate = self.estimate_runtime(exp_type)
            resources = self.calculate_reasonable_resources(exp_type)
            
            print(f"\n{exp_type.upper()} Experiment:")
            print(f"  Resource Request: {resources['cores']} cores, {resources['memory_gb']}GB, {resources['time_hours']}h")
            print(f"  Grid Sizes: {config['grid_sizes']}")
            print(f"  Agent Numbers: {config['n_agents_list']}")
            print(f"  Total Tasks: {estimate['total_tasks']}")
            print(f"  Estimated Runtime: {estimate['parallel_hours']:.1f} hours")
            print(f"  Speedup vs Sequential: {estimate['speedup']:.1f}x")
            print(f"  Will Complete in Time: {'Yes' if estimate['will_fit_in_time_limit'] else 'No'}")
            print(f"  Config: {config_path}")
            print(f"  Script: {script_path}")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("  • Start with 'test' to verify everything works")
        print("  • Use 'small' for initial results (gets scheduled quickly)")
        print("  • Use 'standard' for full publication-quality results")
        print("  • Use 'large' only if you need maximum statistical power")
        print("\nAll configurations use TRUE MPC (no fast mode) for accurate results.")
        print("\nNEW FEATURES:")
        print("  • Multiple grid sizes (10x10 to 40x40)")
        print("  • Multiple agent numbers (2, 3, 4 agents)")
        print("  • Results organized by grid size and agent count")


def main():
    manager = ReasonableResourceManager()
    manager.generate_all_configurations()
    
    print("\n🎯 QUICK START:")
    print("1. Test first:     sbatch scripts/run_test.sh")
    print("2. Small experiment: sbatch scripts/run_small.sh") 
    print("3. Full experiment:  sbatch scripts/run_standard.sh")
    print("\n📊 Monitor: squeue -u $USER")


if __name__ == "__main__":
    main()