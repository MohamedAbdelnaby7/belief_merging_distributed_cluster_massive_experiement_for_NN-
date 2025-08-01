#!/bin/bash
#SBATCH -N 4
#SBATCH -n 128
#SBATCH --mem=256G
#SBATCH -J "BeliefMerging-large"
#SBATCH -p long
#SBATCH -t 110:00:00
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@wpi.edu

# Conservative resource allocation for shared cluster
echo "=========================================="
echo "Belief Merging Experiment - LARGE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: 4"
echo "CPU Cores: 128"
echo "Memory: 256G"
echo "Time Limit: 110 hours"
echo "Partition: long"
echo "Description: Large comprehensive study"
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
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "  Disk space: $(df -h . | tail -1 | awk '{print $4}')"

# Clean up any previous partial runs
find checkpoints/ -name "*.tmp" -delete 2>/dev/null || true
find checkpoints/ -name "*.pkl" -size 0 -delete 2>/dev/null || true

# Start monitoring (lightweight)
(
    while true; do
        echo "$(date): CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% MEM=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2*100}')" >> logs/resource_usage.log
        sleep 300  # Every 5 minutes
    done
) &
MONITOR_PID=$!

# Run experiment with TRUE MPC (no fast mode)
echo "Starting TRUE MPC experiment at $(date)"
echo "WARNING: True MPC is computationally intensive - will take longer but give accurate results"

python3 complete_distributed_experiment.py \
    --config-file configs/large_config.json \
    --max-workers 128 \
    --checkpoint-dir checkpoints \
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
with open('configs/large_config.json', 'r') as f:
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
