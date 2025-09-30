#!/usr/bin/env python3
import pickle
import sys
from pathlib import Path

# Find the most recent pkl file
pkl_files = list(Path('.').glob('kl_focused_results_20250920_*.pkl'))
if not pkl_files:
    print("No kl_focused_results_*.pkl files found!")
    sys.exit(1)

latest_pkl = max(pkl_files, key=lambda x: x.stat().st_mtime)
print(f"Loading: {latest_pkl}")

with open(latest_pkl, 'rb') as f:
    results = pickle.load(f)

# Check interval organization
try:
    for interval_key in ['interval_1', 'interval_10', 'interval_25', 'no_merge', 'immediate_merge']:
        if '20x20' in results and '4_agents' in results['20x20']:
            if 'random' in results['20x20']['4_agents']:
                if interval_key in results['20x20']['4_agents']['random']:
                    trials = results['20x20']['4_agents']['random'][interval_key]['kl_divergence']
                    intervals_found = set(t.get('merge_interval', 'MISSING') for t in trials)
                    print(f"{interval_key}: contains intervals {intervals_found} ({len(trials)} trials)")
except KeyError as e:
    print(f"Key error: {e}")
    print("Available grid keys:", list(results.keys()))
    if '20x20' in results:
        print("Available agent keys:", list(results['20x20'].keys()))