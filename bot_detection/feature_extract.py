from datetime import datetime, timezone
from dateutil import parser
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
import numpy as np
import torch


def parse_note(note):
    """
    Parse the 'note' field and extract plain text.
    :param note: The 'note' field in HTML format
    :return: Extracted plain text string
    """
    # Check if note is empty or not a string
    if not isinstance(note, str) or not note.strip():
        return ""  # Return empty string if note is invalid
    
    try:
        soup = BeautifulSoup(note, "html.parser")
        # Extract plain text
        return soup.get_text(separator=" ").strip()
    except Exception as e:
        print(f"Error parsing note: {note}, Exception: {e}")
        return ""

def calc_activate_days(created_at):
    """
    Calculate the number of days an account has been active.
    :param created_at: Creation time string in ISO8601 format, e.g. "2017-04-25T00:00:00.000+09:00"
    :return: Number of active days
    """
    # Remove extra spaces
    created_at = created_at.strip()
    
    # Parse ISO8601 time
    # create_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%S.%fZ')
    create_date = parser.isoparse(created_at)
    
    # Set to UTC timezone
    create_date = create_date.replace(tzinfo=timezone.utc)
    
    # Crawling date (example)
    crawl_date = datetime.strptime('2024-10-31', '%Y-%m-%d').replace(tzinfo=timezone.utc)
    
    # Calculate time difference
    delta_date = crawl_date - create_date
    
    # Return the difference in days
    return delta_date.days


def extract_profile_features(df,path):
    """
    Extract profile features from a DataFrame.
    :param df: Input pandas DataFrame
    :param path: Path to save the extracted features
    :return: None
    """
    profile_features = []

    for _, row in tqdm(df.iterrows(),total=len(df),desc='Processing profile features'):
        
        acct = row['acct']
        locked = int(row['locked'] == "True")
        bot = int(row['bot'] is False)
        
        discoverable = int(row['discoverable'] == "True")
        created_at = row['created_at']
        avatar = row['avatar']
        followers_count = int(row['followers_count'])
        following_count = int(row['following_count'])
        statuses_count = int(row['statuses_count'])
        fields_data = row['fields']
        uid = row['uid']
        islocal = int(row['islocal'] == "True")
        ismastodon = int(row['ismastodon'] == "True")

        # Feature extraction
        acctname_length = len(acct.split("@")[0])
        active_days = calc_activate_days(created_at)
        note = parse_note(row['note'])
        notes_length = len(note)
        fields_count = len(json.loads(fields_data)) if fields_data else 0

        # Aggregate features as a list of values
        profile_features.append([
            acct,               # String feature
            uid,                # String feature
            bot,                # Numeric feature
            acctname_length,    # Numeric feature
            active_days,        # Numeric feature
            locked,             # Numeric feature
            discoverable,       # Numeric feature
            islocal,            # Numeric feature
            ismastodon,         # Numeric feature
            notes_length,       # Numeric feature
            avatar,             # String feature
            followers_count,    # Numeric feature
            following_count,    # Numeric feature
            statuses_count,     # Numeric feature
            fields_count,       # Numeric feature
            # note,
        ])

    properties = np.array(list(profile_features))

    print('extracting num_properties')
    print('*'*100)
    acctname_length = properties[:,3].astype(float)
    active_days = properties[:,4].astype(float)
    notes_length = properties[:,9].astype(float)
    followers_count = properties[:,11].astype(float)
    following_count = properties[:,12].astype(float)
    statuses_count = properties[:,13].astype(float)
    fields_count = properties[:,14].astype(float)
    
    # Normalize and convert features to tensors
    acctname_length=pd.DataFrame(acctname_length)
    acctname_length=(acctname_length-acctname_length.mean())/acctname_length.std()
    acctname_length=torch.tensor(np.array(acctname_length),dtype=torch.float32)

    notes_length=pd.DataFrame(notes_length)
    notes_length=(notes_length-notes_length.mean())/notes_length.std()
    notes_length=torch.tensor(np.array(notes_length),dtype=torch.float32)
    
    # Fill missing values and normalize active_days
    active_days[np.isnan(active_days)] = 1
    active_days = pd.DataFrame(active_days.astype(np.float32))
    active_days.fillna(int(0))
    active_days = active_days.fillna(int(0)).astype(np.float32)
    active_days = (active_days - active_days.mean()) / active_days.std()
    active_days = torch.tensor(np.array(active_days), dtype=torch.float32)
    
    followers_count=pd.DataFrame(followers_count)
    followers_count=(followers_count-followers_count.mean())/followers_count.std()
    followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)

    following_count=pd.DataFrame(following_count)
    following_count=(following_count-following_count.mean())/following_count.std()
    following_count=torch.tensor(np.array(following_count),dtype=torch.float32)

    statuses_count=pd.DataFrame(statuses_count)
    statuses_count=(statuses_count-statuses_count.mean())/statuses_count.std()
    statuses_count=torch.tensor(np.array(statuses_count),dtype=torch.float32)

    fields_count=pd.DataFrame(fields_count)
    fields_count=(fields_count-fields_count.mean())/fields_count.std()
    fields_count=torch.tensor(np.array(fields_count),dtype=torch.float32)
    
    # Concatenate selected numeric features into a single tensor
    num_properties_tensor=torch.cat([followers_count,active_days,acctname_length,notes_length,following_count,statuses_count],dim=1)
    print('properties[:,0]:',properties[:,0])
    print('+'*100)
    num_properties={
        'user_list':properties[:,0].tolist(),
        'embedding_tensor':num_properties_tensor
    }
    torch.save(num_properties, path+'4k_num_feats.pt')

    print('extracting cat_properties')
    print('*'*100)
    locked = properties[:,5].astype(float)
    discoverable = properties[:,6].astype(float)
    islocal = properties[:,7].astype(float)
    ismastodon = properties[:,8].astype(float)
    
    locked=pd.DataFrame(locked)
    discoverable=pd.DataFrame(discoverable)
    islocal=pd.DataFrame(islocal)
    ismastodon=pd.DataFrame(ismastodon)

    locked_tensor = torch.tensor(np.array(locked), dtype=torch.float)
    discoverable_tensor = torch.tensor(np.array(discoverable), dtype=torch.float)
    islocal_tensor = torch.tensor(np.array(islocal), dtype=torch.float)
    ismastodon_tensor = torch.tensor(np.array(ismastodon), dtype=torch.float)
    
    # Concatenate categorical features into a single tensor
    cat_properties_tensor=torch.cat([locked_tensor,discoverable_tensor,islocal_tensor,ismastodon_tensor],dim=1)
    cat_properties={
        'user_list':properties[:,0].tolist(),
        'embedding_tensor':num_properties_tensor
    }
    torch.save(cat_properties, path+'4k_cat_feats.pt')


# 提取账号的 instance 部分
def extract_instance(acct):
    parts = acct.split("@")
    if len(parts) == 2:
        return parts[1]
    return None



def filter_tweet_embeddings_by_accounts(tweet_feats, accounts_list):
    """
    根据 accounts_list 筛选 tweet_feats 中的 embedding。
    
    参数:
        tweet_feats (dict): {'user_list': [...], 'embedding_tensor': tensor}
        accounts_list (list of str): 要保留的账号列表
        
    返回:
        dict: 筛选后的 {'user_list': [...], 'embedding_tensor': tensor}
    """
    user_list = tweet_feats['user_list']
    embedding_tensor = tweet_feats['embedding_tensor']

    # 构建掩码：哪些用户在 accounts_list 中
    mask = [user in accounts_list for user in user_list]

    # 筛选 embedding 和对应的用户名
    filtered_embeddings = embedding_tensor[mask]
    filtered_users = [user for user, m in zip(user_list, mask) if m]

    # 构造输出结果
    filtered_data = {
        'user_list': filtered_users,
        'embedding_tensor': filtered_embeddings
    }

    print(f"原始 embedding 数量: {len(user_list)}")
    print(f"筛选后 embedding 数量: {len(filtered_users)}")

    return filtered_data


def filter_image_embeddings_by_accounts(tweet_feats, accounts_list):
    """
    根据 accounts_list 筛选 tweet_feats 中的 embedding。
    
    参数:
        tweet_feats (dict): {'user_list': [...], 'embedding_tensor': tensor}
        accounts_list (list of str): 要保留的账号列表
        
    返回:
        dict: 筛选后的 {'user_list': [...], 'embedding_tensor': tensor}
    """
    user_list = image_features['user_list']
    embedding_tensor = image_features['embedding_tensor']

    mask = [user in accounts_list for user in user_list]
    # filter embedding
    filtered_embeddings = embedding_tensor[mask]
    filtered_users = [user for user, m in zip(user_list, mask) if m]

    filtered_data = {
        'user_list': filtered_users,
        'embedding_tensor': filtered_embeddings
    }
    print(f"filtered embedding length: {len(filtered_users)}")

    return filtered_data



filtered_instances = pd.read_csv("../dataset/processed_data/filtered_instances_summary.csv")
target_instances = filtered_instances['instance'].tolist()

path = '../dataset/'
accounts_info = pd.read_csv(path+'accounts_info_60k.csv')
df_accounts = pd.read_csv(path+'accounts_12k_labeled.csv')
accounts_list = df_accounts['acct'].tolist()
accounts_label = df_accounts['label'].tolist()
acct_label = dict(zip(accounts_list,accounts_label))

accounts_info = accounts_info[accounts_info['acct'].isin(accounts_list)]
accounts_info['instance'] = accounts_info['acct'].apply(extract_instance)

print('accounts_info:',len(accounts_info))

# 筛选属于目标 instance 的账号
accounts_info = accounts_info[accounts_info['instance'].isin(target_instances)]
accounts_info['label'] = accounts_info['acct'].map(acct_label)

accounts_info.to_csv(path+'accounts_info_4k.csv')
accounts_label_info = accounts_info[['acct','label']]
accounts_label_info.to_csv(path+'accounts_4k_labeled.csv')



path1 = '../dataset/embeddings/'

extract_profile_features(accounts_info,path1)

# extract tweet text embedding
tweet_feats = torch.load(path1+'ugc_60k_embeddings.pt')
user_list = tweet_feats['user_list']
print(user_list[:5])

filtered_tweet_embeddings = filter_tweet_embeddings_by_accounts(tweet_feats, accounts_info['acct'].tolist())
torch.save(filtered_tweet_embeddings, path1+'4k_tweet_feats.pt')

image_feats = torch.load(path1+'user_image_features.pt')
image_features = {
    'user_list': df_accounts['acct'].tolist(),
    'embedding_tensor': image_feats
}
filtered_image_embeddings = filter_image_embeddings_by_accounts(image_features, accounts_info['acct'].tolist())
torch.save(filtered_image_embeddings, path1+'4k_image_feats.pt')




    
