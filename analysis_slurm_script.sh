#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -J "BeliefAnalysis"
#SBATCH -p short
#SBATCH -t 4:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@wpi.edu

# Advanced Analysis for Massive Belief Merging Dataset
# Designed for high-memory analysis of large-scale experiment results

echo "=========================================="
echo "ADVANCED BELIEF MERGING ANALYSIS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_NTASKS"
echo "Memory: 32GB"
echo "Time Limit: 4 hours"
echo "Analysis Target: Massive distributed experiment dataset"
echo "=========================================="

echo "Navigating to experiment directory..."
cd /home/$USER/belief_merging_NN

# Verify we're in the right place
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la | head -10

# Load environment
echo "Setting up environment..."
cd $SLURM_SUBMIT_DIR

# Load Python module
if module load python/3.10.17/v6xrl7k 2>/dev/null; then
    echo "Loaded Python 3.10.17 module"
elif module load python/3.11.11/hgrhrqx 2>/dev/null; then
    echo "Loaded Python 3.11.11 module"
else
    echo "Using system Python"
fi

# Setup Python path
export PATH=/home/$USER/.local/bin:$PATH
export PYTHONPATH=/home/$USER/.local/lib/python3.10/site-packages:$PYTHONPATH

# Environment optimization for large dataset processing
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0  # For reproducible results

# Pre-analysis checks
echo ""
echo "Pre-analysis system check:"
echo "  Python: $(python3 --version)"
echo "  Working directory: $(pwd)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "  Disk space: $(df -h . | tail -1 | awk '{print $4}')"

# Check for required files
echo ""
echo "Checking required files..."

REQUIRED_FILES=(
    "advanced_cluster_analyzer.py"
    "results/consolidated_results_*.pkl"
)

missing_files=0
for pattern in "${REQUIRED_FILES[@]}"; do
    if ! ls $pattern >/dev/null 2>&1; then
        echo "Missing: $pattern"
        missing_files=$((missing_files + 1))
    else
        echo "Found: $pattern"
    fi
done

if [[ $missing_files -gt 0 ]]; then
    echo "Missing required files. Exiting."
    exit 1
fi

# Count data size
CHECKPOINT_COUNT=$(find checkpoints/ -name "*.pkl" 2>/dev/null | wc -l)
RESULT_FILES=$(find results/ -name "consolidated_results_*.pkl" 2>/dev/null | wc -l)

echo ""
echo "Dataset overview:"
echo "  Checkpoint files: $CHECKPOINT_COUNT"
echo "  Consolidated result files: $RESULT_FILES"

if [[ $CHECKPOINT_COUNT -eq 0 && $RESULT_FILES -eq 0 ]]; then
    echo "No data files found. Exiting."
    exit 1
fi

# Create analysis directories
mkdir -p {results/advanced_analysis,logs}

# Start resource monitoring
(
    while true; do
        echo "$(date '+%H:%M:%S'): Memory=$(free | grep Mem | awk '{printf "%.1fGB", $3/1024/1024}') CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%" >> logs/analysis_resources.log
        sleep 60  # Every minute
    done
) &
MONITOR_PID=$!

echo ""
echo "=========================================="
echo "STARTING ADVANCED ANALYSIS"
echo "=========================================="
echo "Start time: $(date)"
echo "Estimated duration: 30-120 minutes (depending on dataset size)"
echo ""

# Run the advanced analysis
echo "Launching advanced cluster-scale analyzer..."

python3 advanced_cluster_analyzer.py 2>&1 | tee logs/analysis_detailed.log

ANALYSIS_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETED"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $ANALYSIS_EXIT_CODE"

# Check results
if [[ $ANALYSIS_EXIT_CODE -eq 0 ]]; then
    echo "Analysis completed successfully!"
    
    # Count generated files
    ANALYSIS_FILES=$(find results/advanced_analysis/ -type f 2>/dev/null | wc -l)
    PNG_FILES=$(find results/advanced_analysis/ -name "*.png" 2>/dev/null | wc -l)
    
    echo ""
    echo "Analysis Results:"
    echo "  Total files generated: $ANALYSIS_FILES"
    echo "  Visualization files: $PNG_FILES"
    echo "  Analysis directory: results/advanced_analysis/"
    echo ""
    echo "Key files to review:"
    echo "  executive_summary.md - Main insights and recommendations"
    echo "  structured_data.csv - Complete dataset for further analysis"
    echo "  *.png files - Comprehensive visualizations"
    echo ""
    echo "Quick insights:"
    
    # Try to extract quick insights from executive summary
    if [[ -f "results/advanced_analysis/executive_summary.md" ]]; then
        echo "  Best performing strategy:"
        grep "Best Strategy" results/advanced_analysis/executive_summary.md 2>/dev/null | head -1 | sed 's/^/    /'
        echo "  Pattern-specific winners:"
        grep -A 10 "Pattern-Specific Winners" results/advanced_analysis/executive_summary.md 2>/dev/null | tail -n +2 | head -5 | sed 's/^/    /'
    fi
    
else
    echo "Analysis failed with exit code: $ANALYSIS_EXIT_CODE"
    echo "Check logs/analysis_detailed.log for details"
fi

# Resource usage summary
echo ""
echo "Resource Usage Summary:"
if [[ -f "logs/analysis_resources.log" ]]; then
    echo "  Peak memory usage:"
    grep "Memory=" logs/analysis_resources.log | sort -t'=' -k2 -rn | head -1 | sed 's/^/    /'
    echo "  Average CPU usage:"
    grep "CPU=" logs/analysis_resources.log | awk -F'CPU=' '{sum+=$2} END {printf "    %.1f%%\n", sum/NR}' 2>/dev/null || echo "    Could not calculate"
fi

echo ""
echo "Log files:"
echo "  Main output: logs/analysis_${SLURM_JOB_ID}.out"
echo "  Error log: logs/analysis_${SLURM_JOB_ID}.err"
echo "  Detailed log: logs/analysis_detailed.log"
echo "  Resource log: logs/analysis_resources.log"

echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo "1️⃣ Review executive summary:"
echo "   cat results/advanced_analysis/executive_summary.md"
echo ""
echo "2️⃣ View visualizations:"
echo "   ls results/advanced_analysis/*.png"
echo ""
echo "3️⃣ Download results for local viewing:"
echo "   scp -r $USER@turing.wpi.edu:~/belief_merging_NN/results/advanced_analysis ."
echo ""
echo "4️⃣ Use structured data for custom analysis:"
echo "   pandas.read_csv('results/advanced_analysis/structured_data.csv')"
echo ""
echo "=========================================="

exit $ANALYSIS_EXIT_CODE