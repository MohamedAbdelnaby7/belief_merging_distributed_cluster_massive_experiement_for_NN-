#!/bin/bash

# Final Reasonable Deployment for Turing Cluster
# Modified to support multiple grid sizes, agent numbers, and MERGE METHODS
# Conservative resource allocation for shared university cluster
# TRUE MPC (no fast mode) for accurate results

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'Q
PURPLE='\033[0;35m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
header() { echo -e "${PURPLE}[SETUP]${NC} $1"; }

clear
echo "=================================================================="
echo "MULTI-GRID TURING DEPLOYMENT (ENHANCED)"
echo "=================================================================="
echo "Self-contained belief merging experiment with TRUE MPC"
echo "Now supports multiple grid sizes and agent numbers!"
echo "New: Benchmarking multiple merge strategies (Standard KL, Reverse KL, Geometric, Arithmetic)"
echo "NO external dependencies - everything in one file"
echo "=================================================================="

# Step 1: Verify environment
header "1/6 Verifying Turing cluster environment..."

if ! command -v sbatch &> /dev/null; then
    error "SLURM not detected. Are you on the Turing cluster?"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    error "Python 3 not found"
    exit 1
fi

success "Turing cluster environment verified"

# Step 2: Check required files
header "2/6 Checking required experiment files..."

REQUIRED_FILES=(
    "complete_distributed_experiment.py"
    "experiment_analyzer.py"
    "reasonable_turing_allocation.py"
)

# Note: experiment_analyzer.py might be optional if you rely on the built-in collection, 
# but checking it is good practice. I'll allow it to be missing if the main script handles it.
missing_files=()
if [[ ! -f "complete_distributed_experiment.py" ]]; then missing_files+=("complete_distributed_experiment.py"); fi
if [[ ! -f "reasonable_turing_allocation.py" ]]; then missing_files+=("reasonable_turing_allocation.py"); fi

if [[ ${#missing_files[@]} -gt 0 ]]; then
    error "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  $file"
    done
    exit 1
fi

success "Core experiment files found"

# Step 3: Setup Python environment
header "3/6 Setting up Python environment..."

# Load Python module
if module load python/3.10.17/v6xrl7k >/dev/null 2>&1; then
    success "Loaded Python 3.10.17 module"
elif module load python/3.11.11/hgrhrqx >/dev/null 2>&1; then
    success "Loaded Python 3.11.11 module"
elif module load python/3.12.7/zouuiib >/dev/null 2>&1; then
    success "Loaded Python 3.12.7 module"
else
    warning "Could not load Python module, using system Python"
fi

# Install packages with --user (skip virtual environment)
pip install --user --upgrade pip >/dev/null 2>&1
pip install --user numpy scipy matplotlib seaborn pandas psutil >/dev/null 2>&1

# Verify packages are available
python3 -c "import numpy, scipy, matplotlib, seaborn, pandas, psutil" && success "Python packages installed" || error "Package installation failed"

success "Python environment ready"

# Step 4: Create directory structure
header "4/6 Setting up directory structure..."

mkdir -p {checkpoints,results,logs,configs,scripts}
success "Directory structure created"

# Step 5: Generate reasonable configurations
header "5/6 Generating reasonable configurations for shared cluster..."

python3 reasonable_turing_allocation.py

echo ""
echo "RESOURCE ALLOCATION SUMMARY:"
echo "================================"
echo ""
echo "🧪 TEST:      8 cores,  16GB,  2h  (immediate scheduling)"
echo "📊 SMALL:    32 cores,  64GB, 12h  (schedule in hours)"  
echo "🎯 STANDARD: 64 cores, 128GB, 24h  (schedule in ~1 day)"
echo "🚀 LARGE:   128 cores, 256GB, 48h  (schedule in 1-2 days)"
echo ""
echo "All configurations use TRUE MPC (no fast mode) for accuracy"
echo ""
echo "MULTI-GRID SUPPORT:"
echo "================================"
echo "Grid sizes: 10x10, 15x15, 20x20, 25x25, 30x30, 40x40"
echo "Agent numbers: 2, 3, 4 agents"
echo "Results organized by grid_size → agent_count → pattern → strategy"

# Step 6: Select experiment size
header "6/6 Experiment configuration selection..."

echo ""
echo "Select your experiment configuration:"
echo ""
echo "1) 🧪 TEST     - Quick verification (2 grids, 2-3 agents, 5 trials)"
echo "2) 📊 SMALL    - Initial results (2 grids, all agents, 20 trials)"
echo "3) 🎯 STANDARD - Full study (3 grids, all agents, 50 trials)"
echo "4) 🚀 LARGE    - Maximum power (6 grids, all agents, 100 trials)"
echo ""
read -p "Choose configuration (1-4) [default: 2]: " config_choice

case $config_choice in
    1) 
        EXPERIMENT_TYPE="test"
        CORES=8
        MEMORY="16GB"
        TIME="2h"
        TRIALS=5
        GRIDS="2 grid sizes"
        AGENTS="2-3 agents"
        ;;
    3) 
        EXPERIMENT_TYPE="standard"
        CORES=64
        MEMORY="128GB"
        TIME="24h"
        TRIALS=50
        GRIDS="3 grid sizes"
        AGENTS="2-4 agents"
        ;;
    4) 
        EXPERIMENT_TYPE="large"
        CORES=128
        MEMORY="256GB"
        TIME="48h"
        TRIALS=100
        GRIDS="6 grid sizes"
        AGENTS="2-4 agents"
        ;;
    *) 
        EXPERIMENT_TYPE="small"
        CORES=32
        MEMORY="64GB"
        TIME="12h"
        TRIALS=20
        GRIDS="2 grid sizes"
        AGENTS="2-4 agents"
        ;;
esac

info "Selected: $EXPERIMENT_TYPE experiment ($CORES cores, $MEMORY, $TIME)"
info "Configuration: $GRIDS, $AGENTS, $TRIALS trials each"

# Create monitoring script with multi-grid support
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash
echo "Multi-Grid Belief Merging Experiment Monitor"
echo "==========================================="

# Check SLURM job status
echo "SLURM Job Status:"
if squeue -u $USER | grep -q "BeliefMerging"; then
    echo "  🟢 RUNNING"
    squeue -u $USER | grep "BeliefMerging" | head -5
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

# Show breakdown by grid size if checkpoints exist
if [[ $COMPLETED -gt 0 ]]; then
    echo ""
    echo "Progress by Grid Size:"
    for grid in "10x10" "15x15" "20x20" "25x25" "30x30" "40x40"; do
        count=$(find checkpoints/ -name "*grid${grid}*" -name "*.pkl" 2>/dev/null | wc -l)
        if [[ $count -gt 0 ]]; then
            echo "  📊 Grid $grid: $count tasks"
        fi
    done
fi

# Estimate total tasks
if [[ -f "configs/${EXPERIMENT_TYPE}_config.json" ]]; then
    TOTAL_TASKS=$(python3 -c "
import json
with open('configs/${EXPERIMENT_TYPE}_config.json', 'r') as f:
    config = json.load(f)
total = len(config['grid_sizes']) * len(config['n_agents_list']) * len(config['merge_intervals']) * len(config['target_patterns']) * len(config.get('merge_methods', ['standard_kl'])) * config['n_trials']
print(total)
" 2>/dev/null || echo "unknown")
    
    if [[ "$TOTAL_TASKS" != "unknown" ]]; then
        PROGRESS=$(echo "scale=1; $COMPLETED*100/$TOTAL_TASKS" | bc 2>/dev/null || echo "0")
        echo ""
        echo "  📊 Overall Progress: ${PROGRESS}%"
    fi
fi

echo ""

# Show recent log activity
echo "Recent Activity:"
tail -n 3 logs/experiment_*.out 2>/dev/null | sed 's/^/  /' || echo "  No log files yet"

echo ""
echo "Commands:"
echo "  📊 Full status: squeue -u \$USER"
echo "  📝 Live logs: tail -f logs/experiment_*.out"
echo "  🚫 Cancel job: scancel <job_id>"
EOF

chmod +x scripts/monitor.sh

# Create quick status script
cat > scripts/status.sh << 'EOF'
#!/bin/bash
COMPLETED=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
ERRORS=$(find checkpoints/ -name "*_ERROR.txt" 2>/dev/null | wc -l)

# Count by grid size
GRID_10=$(find checkpoints/ -name "*grid10x10*" -name "*.pkl" 2>/dev/null | wc -l)
GRID_20=$(find checkpoints/ -name "*grid20x20*" -name "*.pkl" 2>/dev/null | wc -l)
GRID_30=$(find checkpoints/ -name "*grid30x30*" -name "*.pkl" 2>/dev/null | wc -l)

if squeue -u $USER | grep -q "BeliefMerging"; then
    echo "Status: 🟢 RUNNING | Total: $COMPLETED | Errors: $ERRORS | 10x10: $GRID_10 | 20x20: $GRID_20 | 30x30: $GRID_30"
else
    echo "Status: 🔴 NOT RUNNING | Total: $COMPLETED | Errors: $ERRORS | 10x10: $GRID_10 | 20x20: $GRID_20 | 30x30: $GRID_30"
fi
EOF

chmod +x scripts/status.sh

success "Monitoring tools created"

# Final instructions
echo ""
success "🎉 DEPLOYMENT COMPLETE!"
echo ""
echo "=================================================================="
echo "NEXT STEPS:"
echo "=================================================================="
echo ""
echo "1️⃣  SUBMIT JOB:"
echo "   sbatch scripts/run_${EXPERIMENT_TYPE}.sh"
echo ""
echo "2️⃣  MONITOR PROGRESS:"
echo "   bash scripts/monitor.sh         # Detailed monitoring"
echo "   bash scripts/status.sh          # Quick status"
echo "   squeue -u \$USER                # SLURM queue status"
echo ""
echo "3️⃣  VIEW LOGS:"
echo "   tail -f logs/experiment_*.out   # Live experiment logs"
echo ""
echo "4️⃣  CHECK RESULTS:"
echo "   ls results/analysis/            # Final analysis files"
echo ""
echo "=================================================================="
echo "EXPERIMENT DETAILS:"
echo "=================================================================="
echo ""
echo "📋 Configuration: $EXPERIMENT_TYPE"
echo "⚙️  Resource Request: $CORES cores, $MEMORY RAM, $TIME time limit"
echo "📊 Grid Sizes: $GRIDS"
echo "👥 Agent Numbers: $AGENTS"
echo "🔬 Trials: $TRIALS per configuration"
echo "🧮 TRUE MPC: Full computational accuracy (no fast mode)"
echo ""
echo "⏱️  Expected Timeline:"
echo "   • Queue wait: Minutes to hours (depending on cluster load)"
echo "   • Execution: $TIME with TRUE MPC complexity"
echo "   • Analysis: Automatic upon completion"
echo ""
echo "📁 Key Files:"
echo "   • Main experiment: complete_distributed_experiment.py"
echo "   • Configuration: configs/${EXPERIMENT_TYPE}_config.json"
echo "   • SLURM script: scripts/run_${EXPERIMENT_TYPE}.sh"
echo "   • Checkpoints: checkpoints/grid*_agents*_*.pkl"
echo "   • Results: results/analysis/"
echo ""
echo "=================================================================="
echo "MULTI-GRID COMPUTATIONAL COMPLEXITY:"
echo "=================================================================="
echo ""
echo "🧮 Scaling with Grid Size and Agent Number:"
echo "   • 10x10 grid, 2 agents = 100 states × 25 joint actions"
echo "   • 20x20 grid, 3 agents = 400 states × 125 joint actions"
echo "   • 30x30 grid, 4 agents = 900 states × 625 joint actions"
echo "   • 40x40 grid, 4 agents = 1600 states × 625 joint actions"
echo ""
echo "📊 Total Computational Load:"

# Calculate total tasks based on selection
case $EXPERIMENT_TYPE in
    test) TOTAL_CONFIGS=12 ;;
    small) TOTAL_CONFIGS=48 ;;
    standard) TOTAL_CONFIGS=189 ;;
    large) TOTAL_CONFIGS=486 ;;
esac

# Account for multiple methods (x4: standard, reverse, geometric, arithmetic)
echo "   • Methods per config: 4"
echo "   • Configurations: ~$TOTAL_CONFIGS unique setups"
echo "   • Tasks per config: $TRIALS trials"
echo "   • Total experiments: ~$((TOTAL_CONFIGS * 4 * TRIALS))"
echo ""
echo "This is why we need $CORES cores for reasonable runtime!"
echo ""
echo "=================================================================="
echo "Ready to start? Run: sbatch scripts/run_${EXPERIMENT_TYPE}.sh"
echo "=================================================================="

# Set the selected experiment type in environment
echo "export EXPERIMENT_TYPE=$EXPERIMENT_TYPE" > .experiment_config