import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set global font and style parameters for plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# Load the summary data for all instances
df = pd.read_csv('../dataset/processed_data/instance_summary.csv')

# Filter instances and users according to the following rules:
# num_accounts > 10, num_edges > 10, ratio_label_0 > 0, ratio_label_1 > 0.1
df = df[
    (df['num_accounts'] > 10) &
    (df['num_edges'] > 10) &
    (df['ratio_label_0'] > 0) &
    (df['ratio_label_1'] > 0.1)
]

# Save the filtered summary information for qualified instances
df.to_csv("../../processed_data/filtered_instances_summary.csv", index=False)

# Print the number and a preview of qualified instances
print("Number of qualified instances:", len(df))
print("\nList of qualified instances:")
print(df.head())

# Calculate the number of bots for each instance (bot_count = total accounts * bot ratio)
df['bot_count'] = df['num_accounts'] * df['ratio_label_1']

# Create a figure with dual Y-axes
fig, ax1 = plt.subplots(figsize=(14, 6))

# Draw a bar chart for the total number of accounts per instance
bars = ax1.bar(df['instance'], df['num_accounts'], color='grey', edgecolor='black', linewidth=1.2)
ax1.set_yscale('log')  # Set y-axis to logarithmic scale
ax1.set_xlabel('Instance', weight='bold', fontsize=20)
ax1.set_ylabel('Number of Accounts', weight='bold', fontsize=20)
ax1.tick_params(axis='y')

# Create a second Y-axis for plotting the bot count as a line chart
ax2 = ax1.twinx()
line, = ax2.plot(df['instance'], df['bot_count'], color='blue', marker='o', markersize=6, linestyle='--', linewidth=2, label='Bot Count')
ax2.set_yscale('log')  # Set the second y-axis to logarithmic scale as well
# Alternative: plot log1p(bot_count) if needed
# line, = ax2.plot(df['instance'], np.log1p(df['bot_count']), color='blue', marker='o', markersize=6, linestyle='--', linewidth=2, label='Bot Count')
ax2.set_ylabel('Number of Bots', color='blue', weight='bold', fontsize=20)
ax2.tick_params(axis='y', labelcolor='blue')

# Add grid lines (only for the X-axis)
ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
fig.tight_layout()
fig.autofmt_xdate(rotation=45)  # Auto-format x-axis labels
fig.tight_layout()

# Add legend for both bars and line
lines = [bars, line]
labels = ['Accounts', 'Bots']
ax1.legend(lines, labels, loc='upper right', fontsize=20)
plt.savefig('figs/account_with_bot_line.pdf', dpi=300, bbox_inches='tight')



