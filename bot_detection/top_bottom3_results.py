import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set font and image size
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25

# Create results directory
results_dir = "results/"
os.makedirs('figs', exist_ok=True)

# Define mapping between model names and corresponding result files
model_files = {
    "BotRGCN": "botrgcn_test_results.csv",
    "BECE": "bece_test_results.csv",
    "SGBot": "sgbot_test_results.csv"
}

# Load the file used to filter instances
summary_df = pd.read_csv("../../processed_data/filtered_instances_summary.csv")
bece_test_results = pd.read_csv(os.path.join(results_dir, "bece_test_results.csv"))

instances = bece_test_results['instance_name'].tolist()
summary_df = summary_df[summary_df['instance'].isin(instances)]
top_3_instances = summary_df.sort_values(by='ratio_label_1', ascending=False).head(3)['instance'].tolist()
bottom_3_instances = summary_df.sort_values(by='ratio_label_1', ascending=True).head(3)['instance'].tolist()

selected_instances = top_3_instances + bottom_3_instances  # Select 6 instances in total
print("Selected instances:", selected_instances)

# Load all model results and add model name
all_dfs = []
for model_name, filename in model_files.items():
    file_path = os.path.join(results_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping this model")
        continue
        
    df = pd.read_csv(file_path)
    df['model_name'] = model_name
    df = df[df['instance_name'].isin(selected_instances)]
    all_dfs.append(df)

# Concatenate all model data
combined_df = pd.concat(all_dfs, ignore_index=True)

# Check if required columns exist
required_columns = ['instance_name', 'model_name', 'accuracy', 'precision', 'recall', 'f1_score']
if not all(col in combined_df.columns for col in required_columns):
    raise ValueError("CSV file is missing required columns, please ensure it contains:", required_columns)


# Define hatch styles for each model
hatch_styles = {
    "BotRGCN": "////",   # Slash
    "BECE": "xxxx",    # Cross
    "SGBot": "...."      # Dots
}
model_order = list(model_files.keys())

# Metric display name mapping
metric_mapping = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1-Score',
}

# colors = { 
#     'BotRGCN':  '#b3cde3', 
#     'BECE': '#ccebc5', 
#     'SGBot': '#fbb4ae' 
# }
palette = sns.color_palette("Set2", n_colors=len(model_order))


for metric_key, metric_display in metric_mapping.items():
    

    # Specify order
    model_order = list(model_files.keys())
    instance_order = selected_instances
    num_models = len(model_order)

    # Set category order to ensure consistency in plotting and sorting
    plot_data = combined_df.copy()
    plot_data = plot_data[plot_data['instance_name'].isin(instance_order)]
    plot_data['instance_name'] = pd.Categorical(plot_data['instance_name'], categories=instance_order, ordered=True)
    plot_data['model_name'] = pd.Categorical(plot_data['model_name'], categories=model_order, ordered=True)

    # Re-sort to ensure consistency in plotting
    plot_data = plot_data.sort_values(['instance_name', 'model_name'])
    print('plot_data:',plot_data)

    num_models = len(model_order)
    num_instances = len(instance_order)

    
    # Prepare data: group by instance, each model as a group
    instance_list = sorted(plot_data['instance_name'].unique(), key=lambda x: instance_order.index(x))
    num_instances = len(instance_list)

    # Set bar width
    width = 0.2
    x = np.arange(num_instances)  # X-axis positions

    plt.figure(figsize=(14, 7))

    # Get current axis
    ax = plt.gca()

    # Draw a bar for each model and set hatch style
    for i, model in enumerate(model_order):
        # Extract data for this model
        model_data = plot_data[plot_data['model_name'] == model]
        
        # Ensure consistent sorting
        model_data = model_data.set_index('instance_name').reindex(instance_order).reset_index()
        
        means = model_data[metric_key].values * 100
        
        # Draw bar chart and specify hatch
        rects = ax.bar(
            x + i * width, 
            means, 
            width, 
            label=model,
            hatch=hatch_styles[model],
            # edgecolor='white',
            # linewidth=1.2,
            # edgecolor='black',
            color=palette[i],
            zorder=3
        )

    # Set chart style
    ax.set_xlabel("Instance", fontsize=25, fontweight='bold')
    ax.set_ylabel(metric_display+ " (%)", fontsize=25, fontweight='bold')
    ax.tick_params(labelsize=25)
    ax.set_ylim(0, 100)
    ax.set_xticks(x + width, instance_order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=25)

    # Legend
    if metric_key == 'accuracy':
        ax.legend(title="", frameon=True, fontsize=25, title_fontsize=25, loc='lower right')
    elif metric_key == 'f1_score':
        ax.legend(title="", frameon=True,fontsize=25, title_fontsize=25, loc='upper right')
    else:
        ax.legend(title="", frameon=True,fontsize=25, title_fontsize=25, loc='lower left')
    

    # Grid lines
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, zorder=0)

    # Auto layout and save
    save_path = f"figs1/{metric_key}_bot.pdf"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image saved: {save_path}")


