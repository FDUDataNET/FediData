import xgboost as xgb
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)

def load_data(path, path1, instance_name):
    accounts = pd.read_csv(path+str(instance_name)+'/accounts.csv')
    labels = torch.tensor(accounts['label'].tolist())

    # Only keep nodes with label=0 and 1 for training/evaluation
    valid_idx = torch.where((labels == 0) | (labels == 1))[0]

    # Check if the dataset is large enough for splitting
    label_counts = torch.bincount(labels[valid_idx])
    min_samples_per_class = 4  # At least 4 samples per class are needed for two splits (train 50%, val 25%, test 25%)

    if len(valid_idx) < 8 or any(count < min_samples_per_class for count in label_counts):
        print(f"{instance_name} dataset is too small to split:")
        print(f"- Total valid samples: {len(valid_idx)}")
        print(f"- Class 0 samples: {label_counts[0]}")
        print(f"- Class 1 samples: {label_counts[1]}")
        print("Each class needs at least 4 samples for splitting (train 50%, val 25%, test 25%)")
        return None

    # Load features
    tweets_tensor = torch.load(path1+'4k_tweet_feats.pt')['embedding_tensor']
    image_tensor = torch.load(path1+'4k_image_feats.pt')['embedding_tensor']
    num_prop = torch.load(path1+'4k_num_feats.pt')['embedding_tensor']
    category_prop = torch.load(path1+'4k_num_feats.pt')['embedding_tensor']

    # Concatenate all features
    features = torch.cat([image_tensor, tweets_tensor, num_prop, category_prop], dim=1)
    features = features.numpy()
    labels = labels.numpy()

    # Split within valid_idx
    train_valid, test_valid = train_test_split(
        valid_idx, test_size=0.5, stratify=labels[valid_idx], random_state=42)

    train_idx, val_idx = train_test_split(
        train_valid, test_size=0.5, stratify=labels[train_valid], random_state=42)

    return features, labels, train_idx, val_idx, test_valid

def train_and_test(features, labels, train_idx, val_idx, test_idx, instance_name):
    # Prepare training data
    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_val = features[val_idx]
    y_val = labels[val_idx]
    X_test = features[test_idx]
    y_test = labels[test_idx]

    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train the model
    num_round = 100
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

    # Prediction
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Print results
    print("Test set results:")
    print(f"Accuracy: {accuracy:.4f}, "
          f"Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, "
          f"F1-score: {f1:.4f}, "
          f"AUC: {roc_auc:.4f}")

    # Save results
    results.append({
        'instance_name': instance_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc
    })

    # Save to CSV
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Test results saved to {os.path.abspath(results_path)}")

# Set paths
path = '../../dataset/processed_data/'
path1 = '../../dataset/embeddings/'

# Create result save path
results_path = "../results/xgboost_test_results.csv"
results = []

# Read instance list
intance_data = pd.read_csv(path+"filtered_instances_summary.csv")
instance_names = intance_data['instance'].tolist()

# Iterate over all instances for training and testing
for instance_name in instance_names:
    print(f"\nProcessing instance: {instance_name}")
    
    # Load data
    data = load_data(path, path1, instance_name=instance_name)
    if data is None:
        print(f"Skip instance {instance_name}\n")
        continue
        
    features, labels, train_idx, val_idx, test_idx = data
    
    # Train and test
    print(f"\nStart training and testing {instance_name}...")
    train_and_test(features, labels, train_idx, val_idx, test_idx, instance_name)
    print(f"Finished training and testing {instance_name}\n")

print(f"All instances processed, results saved to {os.path.abspath(results_path)}")