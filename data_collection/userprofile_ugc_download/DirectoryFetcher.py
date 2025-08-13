#!/usr/bin/env python3
"""
Instance Directory Fetcher

Fetches discoverable local users from Mastodon instances' public directories,
enqueues them for BFS processing, and records account info in MongoDB.
"""

import argparse
import random
import re
import signal
import sys
import time
from datetime import datetime, timezone

import requests
from pymongo import MongoClient, errors, ASCENDING

from utils import create_unique_index, rename_key, create_compound_index

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(
    description='Fetch users from Mastodon instances and enqueue for BFS.'
)
parser.add_argument(
    '--id', type=int, required=True,
    help='Worker ID (index into token list)'
)
args = parser.parse_args()
worker_id = args.id

# ----- MongoDB Setup -----
# Remote MongoDB credentials and connection
USERNAME = 'username'
PASSWORD = 'password'
HOST = 'ip_address'
PORT = 'port_number'
REMOTE_URI = f'mongodb://{USERNAME}:{PASSWORD}@{HOST}:{PORT}'
remote_client = MongoClient(REMOTE_URI)
remote_db = remote_client['mastodon']
instances_collection = remote_db['instances']
users_collection = remote_db['users_bfs']
info_collection = remote_db['accounts_info']
relation_collection = remote_db['relation']
ugc_collection = remote_db['ugc']
error_collection = remote_db['error_log']

# Ensure unique indexes
create_unique_index(info_collection, 'uid')
create_unique_index(users_collection, 'acct')
# Optionally create compound indexes for status and layer queries
# create_compound_index(users_collection, 'bfs_status', 'instance_name', 'bfs_layer')
# create_compound_index(users_collection, 'ugc_status', 'instance_name', 'bfs_layer')
# create_compound_index(info_collection, 'processing_status', 'instance_name', 'bfs_layer')

# ----- Load API Token -----
with open('token_list.txt') as f:
    tokens = f.read().splitlines()
HEADERS = {'Authorization': f'Bearer {tokens[worker_id]}'}

# Track the current instance being processed for cleanup
current_processing_instance = None

# ----- Signal Handlers -----
def handle_exit(signum, frame):
    """Reset the processing_status of the current instance on exit."""
    global current_processing_instance
    if current_processing_instance:
        name = current_processing_instance['name']
        print(f"Interrupted: resetting instance '{name}' to pending...")
        instances_collection.update_one(
            {'name': name},
            {'$set': {'processing_status': 'pending'}}
        )
    print("Cleanup complete. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ----- Rate Limit Handling -----
def judge_sleep(res_headers: dict):
    """
    Sleep until the rate-limit reset time if no remaining calls.
    Returns False if slept, True otherwise.
    """
    headers = {k.lower(): v for k, v in res_headers.items()}
    remaining = int(headers.get('x-ratelimit-remaining', 1))
    if remaining <= 0:
        reset_ts = headers.get('x-ratelimit-reset')
        if reset_ts:
            try:
                # Parse ISO timestamp
                if reset_ts.endswith('Z'):
                    reset_ts = reset_ts[:-1] + '+00:00'
                target = datetime.fromisoformat(
                    reset_ts.replace('T', ' ')
                ).replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                sleep_secs = (target - now).total_seconds()
                if sleep_secs > 0:
                    print(f"Rate limit reached. Sleeping until {target} UTC...")
                    time.sleep(sleep_secs)
                    return False
            except Exception as e:
                print(f"Error parsing rate-limit reset time '{reset_ts}': {e}")
    return True

# ----- Error Logging -----
def save_error_log(data_name: str, object_name: str,
                   content: str, res_code='None', error_message=None):
    """Insert an error record into the error_log collection."""
    error_collection.insert_one({
        'loadtime': datetime.now(),
        'data_name': data_name,
        'object': object_name,
        'content': content,
        'response_code': res_code,
        'error_message': str(error_message) if error_message else None
    })

# ----- Directory Fetching -----
def get_users(instance_name: str) -> bool:
    """
    Fetch discoverable local users from an instance's public directory.
    Returns True on success, False on failure.
    """
    url = f'https://{instance_name}/api/v1/directory'
    params = {'limit': 80, 'local': True}
    offset = 0
    retries = 0

    while True:
        try:
            response = requests.get(
                url, headers=HEADERS, params={'limit': 80, 'offset': offset}, timeout=5
            )
            headers = {k.lower(): v for k, v in response.headers.items()}
            judge_sleep(headers)

            if response.status_code == 200:
                users = response.json()
                timestamp = datetime.now()
                save_user_accountinfo_bfs(users, timestamp)
                count = len(users)
                offset += count
                if count < 80:
                    break
            elif response.status_code in (429, 503):
                retries += 1
                time.sleep(random.uniform(0, 5))
                if retries > 5:
                    save_error_log('directory', instance_name,
                                   'Rate or service error', res_code=response.status_code)
                    return False
                continue
            else:
                save_error_log('directory', instance_name,
                               'Unexpected status code', res_code=response.status_code)
                return False
        except requests.exceptions.Timeout:
            retries += 1
            time.sleep(random.uniform(0, 5))
            if retries > 5:
                save_error_log('directory', instance_name, 'Timeout')
                return False
            continue
        except Exception as e:
            save_error_log('directory', instance_name,
                           'Exception during directory fetch', error_message=e)
            return False
    return True

# ----- Save Directory Results -----
def save_user_accountinfo_bfs(users: list, timestamp: datetime):
    """
    Save user accounts to BFS queue and accounts_info collection.
    """
    for item in users:
        rename_key(item, 'id', 'user_id')
        match = re.search(r'https?://([^/]+)/@([^/]+)', item.get('url', ''))
        if not match:
            continue
        instance, username = match.groups()
        uid = f"{username}@{instance}#{item['user_id']}"

        # Prepare BFS queue document
        bfs_doc = {
            'acct': f"{username}@{instance}",
            'uid': uid,
            'user_id': item['user_id'],
            'instance_name': instance,
            'url': item['url'],
            'statuses_count': int(item.get('statuses_count', 0)),
            'followers_count': int(item.get('followers_count', 0)),
            'following_count': int(item.get('following_count', 0)),
            'bfs_status': 'pending',
            'ugc_status': 'pending',
            'bfs_layer': 0
        }
        if bfs_doc['followers_count'] <= 0 and bfs_doc['following_count'] <= 0:
            bfs_doc['bfs_status'] = 'completed'
        if bfs_doc['statuses_count'] <= 0:
            bfs_doc['ugc_status'] = 'completed'
        try:
            users_collection.insert_one(bfs_doc)
        except errors.DuplicateKeyError:
            pass

        # Prepare account info document
        info_doc = item.copy()
        info_doc.update({
            'uid': uid,
            'acct': bfs_doc['acct'],
            'islocal': True,
            'isin_fediverselist': True,
            'local_instance_name': instance,
            'loadtime': timestamp,
            'bfs_layer': 0
        })
        try:
            info_collection.insert_one(info_doc)
        except errors.DuplicateKeyError:
            pass

# ----- Main Loop -----
if __name__ == '__main__':
    while True:
        task = instances_collection.find_one_and_update(
            {'processing_status': 'pending'},
            {'$set': {'processing_status': 'read'}}
        )
        if not task:
            print('No pending instances. Sleeping...')
            time.sleep(60)
            continue

        current_processing_instance = task
        name = task['name']
        print(f"Starting directory fetch for instance: {name}")

        success = get_users(name)
        new_status = 'completed' if success else 'error'
        instances_collection.update_one(
            {'name': name},
            {'$set': {'processing_status': new_status}}
        )
        print(f"Finished directory fetch for {name}: {new_status}")
        current_processing_instance = None
