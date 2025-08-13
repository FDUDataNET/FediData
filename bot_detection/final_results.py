import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

# Results directory
results_dir = "results/"
os.makedirs('figs', exist_ok=True)

# Mapping of model names to their corresponding result file names
model_files = {
    "BotRGCN": "botrgcn_test_results.csv",
    "BECE": "bece_test_results.csv",
    "SGBot": "sgbot_test_results.csv"
}

# Load data for all models and add a model_name column
all_dfs = []
for model_name, filename in model_files.items():
    file_path = os.path.join(results_dir, filename)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping this model")
        continue
    df = pd.read_csv(file_path)
    df['model_name'] = model_name
    all_dfs.append(df)

# Concatenate all model data into a single DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# Check if all required columns are present
required_columns = ['instance_name', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score']
if not all(col in combined_df.columns for col in required_columns):
    raise ValueError("CSV file is missing required columns. Please ensure each file contains:", required_columns)

# Mapping for renaming metrics in plots
metric_mapping = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1-Score',
}
metrics_to_plot = list(metric_mapping.keys())

markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'P']
marker_dict = {model: markers[i] for i, model in enumerate(model_files.keys())}

# Use a rich color palette for the plots
palette = sns.color_palette("Set2", n_colors=len(model_files))

# Plot each metric separately
for metric in metrics_to_plot:
    display_name = metric_mapping[metric]
    plt.figure(figsize=(10, 6))
    
    # Use seaborn barplot for grouped comparison
    ax = sns.barplot(
        x='instance_name',
        y=metric,
        hue='model_name',
        data=combined_df,
        ci=None,  # Do not show confidence intervals
        palette=palette
    )
    
    plt.title(f"{display_name} Across Instances", fontsize=18)
    plt.xlabel("Instance Name", fontsize=14)
    plt.ylabel(display_name, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title="Model", fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"figs/{metric}_comparison.pdf", dpi=300)
    plt.close()
    print(f"Figure saved: figs/{metric}_comparison.pdf")

