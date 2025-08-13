
# Mastodon Data Collection Pipeline

BFS-based data collection for user information, user relationships, and user-generated contents (UGCs) on Mastodon instances.

## Prerequisites

### Dependencies
```bash
pip install -r requirements.txt
```

### Required APIs & Credentials

1. **instances.social API Token**
   - Visit: [instances.social](https://instances.social/api/token)
   - Register for an API token
   - Used for fetching Mastodon instance metadata

2. **MongoDB Database**
   - Set up MongoDB instance (local or remote)
   - Collections used: `instances`, `users_bfs`, `accounts_info`, `relation`, `ugc`, `error_log`
   - Configure connection details in each script

3. **Mastodon Instance Access Tokens**
   - Add tokens to `token_list.txt` (one per line)
   - Get tokens from various Mastodon instances for distributed collection

### Getting Mastodon API Tokens


1. Go to any Mastodon instance (e.g., mastodon.social/settings/applications)
2. Create new application
3. Copy the access token
4. Add to `token_list.txt`



## Usage

### Step 1: Fetch Instance Information
```bash
python MastoListFetcher.py
```
**What it does:**
- Fetches all Mastodon instances from instances.social API
- Stores instance metadata in MongoDB
- Creates `instances_list.txt` file

**Configuration:**
```python
# Edit MastoListFetcher.py
API_TOKEN = "your_instances_social_token"
MONGO_URI = "mongodb://username:password@host:port"
```

### Step 2: Collect Discoverable Users
```bash
python DirectoryFetcher.py --id 0
```
**What it does:**
- Retrieves users with `discoverable=True` from instances
- Enqueues users in BFS queue for network traversal
- Stores user profile information

**Parameters:**
- `--id`: Worker ID (index into token list, 0-based)

### Step 3: BFS Social Network Crawling
```bash
python bfs_worker.py --id 0
```
**What it does:**
- Dequeues pending users from BFS queue
- Crawls followers/following relationships
- Stores social network connections
- **Note:** User IDs are as seen from the crawling instance

**Key Features:**
- Distributed crawling using multiple tokens
- Automatic rate limiting and error handling
- BFS traversal ensures comprehensive network coverage

### Step 4: Resolve True User IDs
```bash
python get_local_id.py --id 0
```
**What it does:**
- Looks up each user's true local ID on their home instance
- Corrects ID mapping for cross-instance references
- Re-enqueues users with correct IDs

### Step 5: Collect User Posts (UGC)
```bash
python ugc_worker.py --id 0
```
**What it does:**
- Crawls posts (toots) from users in the BFS queue
- Stores post content, metadata, and media information
- Handles pagination for users with many posts


## Output Data Structure

### Collections Created:
- **instances**: Mastodon instance metadata
- **users_bfs**: BFS queue for user discovery
- **accounts_info**: User profile information
- **relation**: Follower/following relationships  
- **ugc**: User posts and content
- **error_log**: Failed requests and errors
