import pandas as pd
import os
import torch

# Input file paths
ACCOUNTS_FILE = "../dataset/accounts_12k_labeled.csv"
EDGES_FILE = "../dataset/edges_60k.csv"

# Output directory
OUTPUT_DIR = "../dataset/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract instance from account string (e.g., user@instance -> instance)
def get_instance(acct):
    return acct.split("@")[-1]

# Load account data
df_accounts = pd.read_csv(ACCOUNTS_FILE)
df_accounts['instance'] = df_accounts['acct'].apply(get_instance)

# Build mapping from account to instance
acct_to_instance = dict(zip(df_accounts['acct'], df_accounts['instance']))

# Group accounts by instance
account_groups = df_accounts.groupby('instance')

summary_list = []

print("Start processing accounts...")
for instance, group in account_groups:
    print(f"\nProcessing instance: {instance}")

    # Create directory for each instance
    instance_dir = os.path.join(OUTPUT_DIR, instance)
    os.makedirs(instance_dir, exist_ok=True)

    # Build mapping from account to local ID within this instance
    acct_list = group['acct'].tolist()
    acct_to_id = {acct: idx for idx, acct in enumerate(acct_list)}

    # Add 'id' column and save accounts.csv for this instance
    group['id'] = group['acct'].map(acct_to_id)
    account_save_path = os.path.join(instance_dir, "accounts.csv")

    try:
        group[['id', 'acct', 'label']].to_csv(account_save_path, index=False)
        print(f"Saved accounts.csv to {account_save_path}")
    except Exception as e:
        print(f"Failed to save accounts.csv: {e}")

    # Calculate label distribution for summary
    total = len(group)
    count_label_0 = group[group['label'] == 0].shape[0]
    count_label_1 = group[group['label'] == 1].shape[0]
    ratio_0 = count_label_0 / total if total > 0 else 0
    ratio_1 = count_label_1 / total if total > 0 else 0

    summary_list.append({
        'instance': instance,
        'num_accounts': total,
        'ratio_label_0': round(ratio_0, 4),
        'ratio_label_1': round(ratio_1, 4),
        'num_edges': 0
    })

# Create summary DataFrame
df_summary = pd.DataFrame(summary_list)

print("Start processing edges...")
df_edges = pd.read_csv(EDGES_FILE)

# Filter edges where both source and target are in the same instance
def is_same_instance(row):
    src_inst = acct_to_instance.get(row['source'])
    tgt_inst = acct_to_instance.get(row['target'])
    return src_inst == tgt_inst and src_inst is not None

same_instance_edges = df_edges[df_edges.apply(is_same_instance, axis=1)]

# Group edges by the instance of the source account
edge_groups = same_instance_edges.groupby(lambda x: acct_to_instance.get(same_instance_edges.loc[x, 'source']))

for instance, group in edge_groups:
    if pd.isna(instance):
        continue

    print(f"\nProcessing edges for instance: {instance}")
    instance_dir = os.path.join(OUTPUT_DIR, instance)

    # Load account mapping for this instance
    account_file = os.path.join(instance_dir, "accounts.csv")
    if not os.path.exists(account_file):
        print(f"No accounts.csv found for {instance}, skipping edges")
        continue

    df_acct = pd.read_csv(account_file)
    acct_to_id = dict(zip(df_acct['acct'], df_acct['id']))

    # Map source/target accounts to their local IDs
    def map_acct(x):
        return acct_to_id.get(x, -1)

    group['source_id'] = group['source'].apply(map_acct)
    group['target_id'] = group['target'].apply(map_acct)

    # Remove invalid edges (where mapping failed)
    valid_edges = group[(group['source_id'] != -1) & (group['target_id'] != -1)]

    print(f"[{instance}] Total edges before filtering: {len(group)}")
    print(f"[{instance}] Valid edges after filtering: {len(valid_edges)}")

    if len(valid_edges) == 0:
        print(f"No valid edges for {instance}, skipping")
        continue

    # Save valid edges to edges.csv
    edge_save_path = os.path.join(instance_dir, "edges.csv")
    try:
        valid_edges.to_csv(edge_save_path, index=False)
        print(f"Saved edges.csv to {edge_save_path}")
    except Exception as e:
        print(f"Failed to save edges.csv: {e}")

    # Save edge_index.pt (PyTorch tensor of edge indices)
    edge_tensor = torch.tensor(valid_edges[['source_id', 'target_id']].values, dtype=torch.long).t()
    edge_tensor_save_path = os.path.join(instance_dir, "edge_index.pt")
    try:
        torch.save(edge_tensor, edge_tensor_save_path)
        print(f"Saved edge_index.pt to {edge_tensor_save_path}")
    except Exception as e:
        print(f"Failed to save edge_index.pt: {e}")

    # Save edge_type.pt if 'relation' column exists (edge types)
    if 'relation' in valid_edges.columns:
        valid_edges = valid_edges.dropna(subset=['relation'])
        unique_relations = valid_edges['relation'].unique()

        if len(unique_relations) == 0:
            print(f"No valid relations found for {instance}, skipping edge_type.pt")
        else:
            relation_to_id = {rel: i for i, rel in enumerate(unique_relations)}
            valid_edges['relation_id'] = valid_edges['relation'].map(relation_to_id)

            edge_type_tensor = torch.tensor(valid_edges['relation_id'].values, dtype=torch.long)
            edge_type_save_path = os.path.join(instance_dir, "edge_type.pt")
            try:
                torch.save(edge_type_tensor, edge_type_save_path)
                print(f"Saved edge_type.tensor to {edge_type_save_path}")
            except Exception as e:
                print(f"Failed to save edge_type.tensor: {e}")
    else:
        print(f"No 'relation' column in edges for {instance}")

    # Update summary with number of valid edges
    df_summary.loc[df_summary['instance'] == instance, 'num_edges'] = len(valid_edges)

# Save summary CSV for all instances
summary_output_path = os.path.join(OUTPUT_DIR, "instance_summary.csv")
df_summary.to_csv(summary_output_path, index=False)
print(f"Summary saved to {summary_output_path}")