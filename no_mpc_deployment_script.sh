#!/bin/bash
# Simplified Belief Merging Experiment Deployment
# Focus: KL-divergence optimization vs ground truth

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

clear
echo "=================================================================="
echo "SIMPLIFIED BELIEF MERGING EXPERIMENT DEPLOYMENT"
echo "=================================================================="
echo "Focus: Testing KL-divergence optimization against ground truth"
echo "No MPC complexity - pure belief merging comparison"
echo "Robust checkpointing and comprehensive visualization"
echo "=================================================================="

# Check environment
info "Checking environment..."

if ! command -v python3 &> /dev/null; then
    error "Python 3 not found"
    exit 1
fi

success "Environment verified"

# Setup directories
info "Setting up directories..."
mkdir -p {checkpoints,results,logs,configs,scripts}
success "Directory structure created"

# Create configuration files
info "Creating experiment configurations..."

# Test configuration
cat > configs/test_config.json << 'EOF'
{
    "grid_sizes": [[10, 10], [15, 15]],
    "n_agents_list": [3, 4],
    "alpha": 0.1,
    "beta": 0.2,
    "n_trials": 10,
    "max_steps": 100,
    "merge_intervals": [5, 10, "inf"],
    "target_patterns": ["random"],
    "checkpoint_dir": "checkpoints"
}
EOF

# Standard configuration
cat > configs/standard_config.json << 'EOF'
{
    "grid_sizes": [[10, 10], [20, 20], [30, 30]],
    "n_agents_list": [2, 3, 4],
    "alpha": 0.1,
    "beta": 0.2,
    "n_trials": 50,
    "max_steps": 200,
    "merge_intervals": [5, 10, 15, float("inf")],
    "target_patterns": ["random"],
    "checkpoint_dir": "checkpoints"
}
EOF

# Large configuration
cat > configs/large_config.json << 'EOF'
{
    "grid_sizes": [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]],
    "n_agents_list": [2, 3, 4, 5],
    "alpha": 0.1,
    "beta": 0.2,
    "n_trials": 100,
    "max_steps": 300,
    "merge_intervals": [5, 10, 15, 25, 50, 100, "inf"],
    "target_patterns": ["random"],
    "checkpoint_dir": "checkpoints"
}
EOF

success "Configuration files created"

# Create SLURM scripts
info "Creating SLURM execution scripts..."

# Test script
cat > scripts/run_test.sh << 'EOF'
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -J "BeliefTest"
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "Starting simplified belief merging test experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

cd $SLURM_SUBMIT_DIR

# Load Python environment
module load python/3.10.17/v6xrl7k 2>/dev/null || echo "Using system Python"

# Run experiment
python3 simplified_belief_experiment.py \
    --config-file configs/test_config.json \
    --max-workers 8 \
    --checkpoint-dir checkpoints \
    --results-dir results

echo "Test experiment completed at $(date)"
EOF

# Standard script
cat > scripts/run_standard.sh << 'EOF'
#!/bin/bash
#SBATCH -N 2
#SBATCH -n 32
#SBATCH --mem=64G
#SBATCH -J "BeliefStandard"
#SBATCH -p long
#SBATCH -t 12:00:00
#SBATCH --output=logs/standard_%j.out
#SBATCH --error=logs/standard_%j.err

echo "Starting simplified belief merging standard experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

cd $SLURM_SUBMIT_DIR

# Load Python environment
module load python/3.10.17/v6xrl7k 2>/dev/null || echo "Using system Python"

# Run experiment
python3 simplified_belief_experiment.py \
    --config-file configs/standard_config.json \
    --max-workers 32 \
    --checkpoint-dir checkpoints \
    --results-dir results

echo "Standard experiment completed at $(date)"
EOF

# Large script
cat > scripts/run_large.sh << 'EOF'
#!/bin/bash
#SBATCH -N 4
#SBATCH -n 150
#SBATCH --mem=128G
#SBATCH -J "BeliefLarge"
#SBATCH -p long
#SBATCH -t 250:00:00
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err

echo "Starting simplified belief merging large experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

cd $SLURM_SUBMIT_DIR

# Load Python environment
module load python/3.10.17/v6xrl7k 2>/dev/null || echo "Using system Python"

# Run experiment
python3 simplified_belief_experiment.py \
    --config-file configs/large_config.json \
    --max-workers 64 \
    --checkpoint-dir checkpoints \
    --results-dir results

echo "Large experiment completed at $(date)"
EOF

chmod +x scripts/*.sh
success "SLURM scripts created"

# Create monitoring tools
info "Creating monitoring tools..."

cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
echo "Simplified Belief Merging Experiment Monitor"
echo "==========================================="

# Check SLURM job status
echo "SLURM Job Status:"
if squeue -u $USER | grep -q "Belief"; then
    echo "  🟢 RUNNING"
    squeue -u $USER | grep "Belief"
else
    echo "  🔴 NOT RUNNING"
fi

echo ""

# Count progress
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

echo "Task Progress:"
echo "  ✅ Completed: $COMPLETED"
echo "  ❌ Errors: $ERRORS"

# Show breakdown by strategy if checkpoints exist
if [[ $COMPLETED -gt 0 ]]; then
    echo ""
    echo "Progress by Strategy:"
    KL_COUNT=$(find checkpoints/ -name "*kl_divergence*" -name "*.pkl" 2>/dev/null | wc -l)
    WEIGHTED_COUNT=$(find checkpoints/ -name "*weighted_average*" -name "*.pkl" 2>/dev/null | wc -l)
    PRODUCT_COUNT=$(find checkpoints/ -name "*product*" -name "*.pkl" 2>/dev/null | wc -l)
    
    echo "  📊 KL Divergence: $KL_COUNT tasks"
    echo "  📊 Weighted Average: $WEIGHTED_COUNT tasks"
    echo "  📊 Product Rule: $PRODUCT_COUNT tasks"
fi

echo ""
echo "Commands:"
echo "  📊 Full status: squeue -u \$USER"
echo "  📝 Live logs: tail -f logs/*.out"
echo "  🚫 Cancel job: scancel <job_id>"
EOF

chmod +x scripts/monitor.sh

cat > scripts/status.sh << 'EOF'
#!/bin/bash
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

if squeue -u $USER | grep -q "Belief"; then
    echo "Status: 🟢 RUNNING | Completed: $COMPLETED | Errors: $ERRORS"
else
    echo "Status: 🔴 NOT RUNNING | Completed: $COMPLETED | Errors: $ERRORS"
fi
EOF

chmod +x scripts/status.sh

# Create local run script for testing
cat > scripts/run_local.sh << 'EOF'
#!/bin/bash
echo "Running simplified belief merging experiment locally..."

python3 simplified_belief_experiment.py \
    --config-file configs/test_config.json \
    --max-workers 4 \
    --checkpoint-dir checkpoints \
    --results-dir results

echo "Local run completed!"
EOF

chmod +x scripts/run_local.sh

success "Monitoring tools created"

# Create analysis script
info "Creating analysis script..."

cat > scripts/analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
Quick analysis script for belief merging results
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results():
    results_dir = Path("results")
    
    # Find latest results file
    result_files = list(results_dir.glob("consolidated_results_*.pkl"))
    if not result_files:
        print("No results found!")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Analyzing: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        results = pickle.load(f)
    
    # Convert to DataFrame
    data_rows = []
    for grid_key, grid_results in results.items():
        for agent_key, agent_results in grid_results.items():
            n_agents = int(agent_key.split('_')[0])
            for pattern, pattern_results in agent_results.items():
                for strategy_key, trials in pattern_results.items():
                    for trial in trials:
                        data_rows.append({
                            'grid_size': grid_key,
                            'n_agents': n_agents,
                            'strategy': trial['merge_strategy'],
                            'kl_to_truth': trial['avg_kl_to_truth'],
                            'target_prob': trial['avg_target_prob_merged'],
                            'prediction_error': trial['prediction_error']
                        })
    
    df = pd.DataFrame(data_rows)
    
    # Quick summary
    print("\nQUICK RESULTS SUMMARY:")
    print("="*50)
    
    strategy_performance = df.groupby('strategy')['kl_to_truth'].agg(['mean', 'std', 'count'])
    print("\nKL Divergence to Ground Truth:")
    print(strategy_performance)
    
    # Best strategy
    best_strategy = strategy_performance['mean'].idxmin()
    print(f"\nBest Strategy: {best_strategy}")
    
    # Statistical comparison
    if 'kl_divergence' in df['strategy'].values and 'weighted_average' in df['strategy'].values:
        kl_data = df[df['strategy'] == 'kl_divergence']['kl_to_truth']
        weighted_data = df[df['strategy'] == 'weighted_average']['kl_to_truth']
        
        improvement = (weighted_data.mean() - kl_data.mean()) / weighted_data.mean() * 100
        print(f"\nKL-divergence vs Weighted Average:")
        print(f"  Improvement: {improvement:.1f}%")
        
        from scipy.stats import mannwhitneyu
        _, p_val = mannwhitneyu(kl_data, weighted_data, alternative='less')
        print(f"  Statistical significance: p={p_val:.4f}")
    
    print(f"\nTotal trials analyzed: {len(df)}")
    print("Full visualizations available in results/analysis/")

if __name__ == "__main__":
    analyze_results()
EOF

chmod +x scripts/analyze_results.py

success "Analysis script created"

# Final setup
info "Performing final setup..."

# Install required packages
pip install --user numpy scipy matplotlib seaborn pandas >/dev/null 2>&1

success "Setup complete!"

# Display usage instructions
echo ""
echo "=================================================================="
echo "DEPLOYMENT COMPLETE - READY TO RUN!"
echo "=================================================================="
echo ""
echo "📋 EXPERIMENT OVERVIEW:"
echo "  • Focus: KL-divergence optimization vs ground truth"
echo "  • No MPC complexity - clean belief merging comparison"
echo "  • Robust checkpointing prevents data loss"
echo "  • Comprehensive visualizations included"
echo ""
echo "🚀 QUICK START OPTIONS:"
echo ""
echo "1️⃣ LOCAL TESTING (recommended first):"
echo "   bash scripts/run_local.sh"
echo ""
echo "2️⃣ CLUSTER TESTING:"
echo "   sbatch scripts/run_test.sh"
echo ""
echo "3️⃣ FULL EXPERIMENT:"
echo "   sbatch scripts/run_standard.sh"
echo ""
echo "4️⃣ LARGE SCALE:"
echo "   sbatch scripts/run_large.sh"
echo ""
echo "📊 MONITORING:"
echo "   bash scripts/monitor.sh     # Detailed status"
echo "   bash scripts/status.sh      # Quick status"
echo "   squeue -u \$USER             # SLURM queue"
echo ""
echo "📈 ANALYSIS:"
echo "   python3 scripts/analyze_results.py  # Quick summary"
echo "   # Full analysis runs automatically after experiment"
echo ""
echo "=================================================================="
echo "EXPERIMENT CONFIGURATIONS:"
echo "=================================================================="
echo ""
echo "🧪 TEST (2h, 8 cores):"
echo "   • 2 grid sizes, 2 agent numbers"
echo "   • 10 trials per configuration"
echo "   • Quick verification"
echo ""
echo "📊 STANDARD (12h, 32 cores):"
echo "   • 3 grid sizes, 3 agent numbers"
echo "   • 50 trials per configuration"
echo "   • Publication-quality results"
echo ""
echo "🚀 LARGE (24h, 64 cores):"
echo "   • 4 grid sizes, 4 agent numbers"
echo "   • 100 trials per configuration"
echo "   • Maximum statistical power"
echo ""
echo "=================================================================="
echo "KEY FEATURES:"
echo "=================================================================="
echo ""
echo "✅ Simplified Focus:"
echo "   • Pure belief merging comparison (no MPC complexity)"
echo "   • Direct test of your KL-divergence optimization"
echo "   • Ground truth computed from all agent observations"
echo ""
echo "✅ Robust Implementation:"
echo "   • Atomic checkpointing prevents data corruption"
echo "   • Automatic resume from interruptions"
echo "   • Consistent seed generation across sessions"
echo ""
echo "✅ Comprehensive Analysis:"
echo "   • Statistical significance testing"
echo "   • Performance scaling analysis"
echo "   • Time series evolution plots"
echo "   • Publication-ready visualizations"
echo ""
echo "✅ Your Exact Algorithm:"
echo "   • No modifications to optimization parameters"
echo "   • No fallback approximations"
echo "   • Clean test of KL-divergence minimization"
echo ""
echo "=================================================================="
echo "Ready to start? Try: bash scripts/run_local.sh"
echo "=================================================================="