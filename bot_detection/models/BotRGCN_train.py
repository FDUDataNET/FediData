from BotRGCN import BotRGCN
import torch
from torch import nn

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch import nn

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def load_data(path,path1,instance_name):
    accounts = pd.read_csv(path+str(instance_name)+'/accounts.csv')
    labels=torch.tensor(accounts['label'].tolist())

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

    # Split within valid_idx
    train_valid, test_valid = train_test_split(
        valid_idx, test_size=0.5, stratify=labels[valid_idx], random_state=42)

    train_idx, val_idx = train_test_split(
        train_valid, test_size=0.5, stratify=labels[train_valid], random_state=42)

    # Convert to tensor
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_valid)
    
    tweets_tensor = torch.load(path1+'4k_tweet_feats.pt')
    tweets_tensor = tweets_tensor['embedding_tensor']

    # image_tensor=torch.load(path1+'image_tensor.pt')
    image_tensor = torch.load(path1+'4k_image_feats.pt')
    image_tensor = image_tensor['embedding_tensor']

    num_prop = torch.load(path1+'4k_num_feats.pt')
    num_prop = num_prop['embedding_tensor']
    category_prop = torch.load(path1+'4k_cat_feats.pt')
    category_prop = category_prop['embedding_tensor']
    edge_index = torch.load(path+str(instance_name)+'/edge_index.pt')
    edge_type = torch.load(path+str(instance_name)+'/edge_type.pt')
    # edge_type = torch.unsqueeze(edge_type, dim=1)
    print('image_tensor:',image_tensor.shape)
    print('tweets_tensor:',tweets_tensor.shape)
    print('num_prop:',num_prop.shape)
    print('category_prop:',category_prop.shape)
    print('edge_index:',edge_index.shape)
    print('edge_type:',edge_type.shape)
    print('labels:',labels.shape,labels.unique())
    print('train_idx:',train_idx.shape)
    print('val_idx:',val_idx.shape)
    print('test_idx:',test_idx.shape)

    max_edge_type = edge_type.max().item()
    print("Max edge type:", max_edge_type)
    print("Number of unique types:", edge_type.unique().size(0))
   
    return image_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx


def train(epoch):
    model.train()
    output = model(image_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_train = loss_fn(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))
    return acc_train, loss_train

def test(instance_name):
    model.eval()
    output = model(image_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_test = loss_fn(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])

    # Convert output and labels to numpy arrays
    probs = torch.softmax(output, dim=1)[:, 1].cpu().detach().numpy()  # Probabilities for AUC
    preds = output.argmax(dim=1).cpu().detach().numpy()
    labels_cpu = labels.cpu().numpy()

    # Extract predictions and true values for test split
    y_true = labels_cpu[test_idx]
    y_pred = preds[test_idx]
    y_score = probs[test_idx]

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Print results
    print("Test set results:")
    print(f"Accuracy: {acc_test.item():.4f}, "
          f"Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, "
          f"F1-score: {f1:.4f}, "
          f"AUC: {roc_auc:.4f}")

    # Save results
    results.append({
        'instance_name': instance_name,
        'accuracy': acc_test.item(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc
    })

    # Save to CSV
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"Test results saved to {os.path.abspath(results_path)}")


path = '../../dataset/processed_data/'
path1 = '../../dataset/embeddings/'

results_path = "../results/botrgcn_test_results.csv"
results = []

device = 'cpu'
embedding_size, dropout, lr, weight_decay = 32, 0.1, 1e-2, 5e-2
epochs = 200

intance_data = pd.read_csv(path+"filtered_instances_summary.csv")
instance_names = intance_data['instance'].tolist()

# model training
for instance_name in instance_names:
    print(f"\n Processing instance: {instance_name}")
    
    data = load_data(path, path1, instance_name=instance_name)
    if data is None:
        print(f"pass {instance_name} instance\n")
        continue
        
    image_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = data
    
    # model initialization
    model = BotRGCN(cat_prop_size=6, embedding_dimension=embedding_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.apply(init_weights)
    
    print(f"start training {instance_name}...")
    for epoch in range(epochs):
        train(epoch)
    
    print(f"\n start testing {instance_name}...")
    test(instance_name)

