import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.0) 

# Define the four main methods and their colors
METHODS = {
    'standard_kl': 'Standard KL\n(Arithmetic)',
    'reverse_kl': 'Reverse KL\n(Geometric)', 
    'arithmetic_mean': 'Arithmetic Mean',
    'geometric_mean': 'Geometric Mean',
    'unknown': 'Unknown'
}

COLORS = {
    'Standard KL\n(Arithmetic)': '#3498db',  # Blue
    'Reverse KL\n(Geometric)': '#e74c3c',    # Red
    'Arithmetic Mean': '#95a5a6',             # Gray
    'Geometric Mean': '#2ecc71',              # Green
    'Unknown': '#333333'
}

def load_and_process_data(results_dir="checkpoints"): # CHANGED DEFAULT TO CHECKPOINTS
    print(f"Scanning {results_dir} for individual trial files...")
    checkpoint_path = Path(results_dir)
    
    # Look for all .pkl files, excluding any consolidated ones if they exist in this folder
    files = [f for f in checkpoint_path.glob("*.pkl") if "consolidated" not in f.name]
    
    if not files:
        print(f"No trial files found in {results_dir}.")
        return pd.DataFrame()

    print(f"Found {len(files)} files. Processing...")

    data = []
    
    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'rb') as f:
                trial = pickle.load(f)
                
                # Determine method from filename
                method_raw = "unknown"
                fname = file_path.stem
                
                if "geometric_mean" in fname: method_raw = 'geometric_mean'
                elif "arithmetic_mean" in fname: method_raw = 'arithmetic_mean'
                elif "reverse_kl" in fname: method_raw = 'reverse_kl'
                elif "standard_kl" in fname: method_raw = 'standard_kl'
                # Fallback for older naming
                elif "geometric" in fname: method_raw = 'geometric_mean'
                elif "arithmetic" in fname: method_raw = 'arithmetic_mean'
                
                # Check internal metadata if filename fails
                if method_raw == "unknown":
                    if isinstance(trial, dict) and 'task_metadata' in trial:
                         method_raw = trial['task_metadata'].get('merge_method', 'unknown')
                
                # Map to display name
                method_name = METHODS.get(method_raw, 'Unknown')
                
                # Skip if truly unknown (likely garbage file)
                if method_name == 'Unknown':
                    # print(f"Skipping unknown method file: {file_path.name}")
                    continue
                
                # Extract metrics
                found = trial.get('target_found', False)
                steps = trial.get('first_discovery_step', 1000) # Cap at max steps
                
                # Handle KL
                kl = 0
                if 'time_series' in trial:
                    kl_series = trial['time_series'].get('kl_divergence_to_truth', [0])
                    if kl_series:
                        kl = np.mean(kl_series)
                elif 'avg_kl_to_truth' in trial:
                    kl = trial['avg_kl_to_truth']
                
                data.append({
                    'Method': method_name,
                    'Success': 1 if found else 0,
                    'Time to Discovery': steps,
                    'KL Divergence': kl
                })
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(files)} files...")

    df = pd.DataFrame(data)
    print(f"Successfully loaded {len(df)} valid trials.")
    return df

def plot_grouped_results(df):
    if df.empty:
        print("No valid data to plot.")
        return

    # Filter only for the 4 methods we care about
    valid_methods = [m for m in list(METHODS.values()) if m != 'Unknown']
    df = df[df['Method'].isin(valid_methods)]
    
    if df.empty:
        print("No data found for the target methods.")
        return

    # Ensure output directory exists
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # 1. Success Rate (Bar Chart with Error Bars)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x='Method', y='Success', palette=COLORS, capsize=.1, errorbar=('ci', 95))
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)

    plt.title("Target Discovery Success Rate\n(Aggregated across all grids & agents)", fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Success Rate")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig('results/grouped_success_rate.png')
    plt.close()

    # 2. Time to Discovery (Box Plot + Strip Plot)
    plt.figure(figsize=(12, 8))
    success_only = df[df['Success'] == 1]
    
    if not success_only.empty:
        sns.boxplot(data=success_only, x='Method', y='Time to Discovery', palette=COLORS, showfliers=False)
        sns.stripplot(data=success_only, x='Method', y='Time to Discovery', color='black', alpha=0.1, jitter=True)
        
        plt.title("Time to Find Target (Successful Trials Only)\n(Lower is Better)", fontweight='bold')
        plt.ylabel("Steps")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig('results/grouped_time_to_discovery_dist.png')
        plt.close()

        # 3. Time to Discovery (Bar)
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=success_only, x='Method', y='Time to Discovery', palette=COLORS, capsize=.1)
        for i in ax.containers:
            ax.bar_label(i, fmt='%.0f', padding=3)
            
        plt.title("Average Time to Find Target\n(Aggregated)", fontweight='bold')
        plt.ylabel("Average Steps")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig('results/grouped_time_to_discovery_bar.png')
        plt.close()

    # 4. KL Divergence (Bar Chart)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x='Method', y='KL Divergence', palette=COLORS, capsize=.1)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.title("Average KL Divergence to Truth\n(Lower is Better)", fontweight='bold')
    plt.ylabel("KL Divergence")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig('results/grouped_kl_divergence.png')
    plt.close()

    print("Aggregated plots saved to results/ directory.")

if __name__ == "__main__":
    # Check if checkpoints dir exists
    if not Path("checkpoints").exists():
        print("Error: 'checkpoints' directory not found.")
    else:
        df = load_and_process_data("checkpoints")
        plot_grouped_results(df)