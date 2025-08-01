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
