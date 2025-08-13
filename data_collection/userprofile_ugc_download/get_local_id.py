#!/usr/bin/env python3
"""
Mastodon Local ID Resolver

Fetches the true local account ID for non-local users, updates processing status,
and handles rate limits and errors in MongoDB.
"""

import argparse
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from utils import (
    create_unique_index,
    judge_sleep,
    judge_api_islimit,
    rename_key
)

# ----- Global Variables -----
current_processing_user = None
limit_dict = {}
limit_set = set()

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(
    description='Resolve local account IDs for Mastodon users.'
)
parser.add_argument('--id', type=int, required=True,
                    help='Worker ID (index into token list)')
parser.add_argument('--priority', type=int, default=0,
                    help='Worker priority affecting instance selection')
args = parser.parse_args()
worker_id = args.id
priority = args.priority

# ----- MongoDB Connections -----
# Local error log
local_client = MongoClient('mongodb://localhost:27018/')
local_db = local_client['mastodon']
local_error_collection = local_db['error_log']

# Remote data
REMOTE_URI = (
    'mongodb://admin:FudanSonicDataNET2024!!!CGGMWL'
    '@128.105.145.239:27018/'
)
remote_client = MongoClient(REMOTE_URI)
remote_db = remote_client['mastodon']
users_collection = remote_db['users_bfs']
info_collection = remote_db['accounts_info']

# Ensure unique indexes
create_unique_index(info_collection, 'uid')
create_unique_index(users_collection, 'acct')

# ----- Load Instances and Tokens -----
instances_list = set()
with open('instances_list.txt') as f:
    for line in f:
        instances_list.add(line.strip())

with open('token_list.txt') as f:
    tokens = f.read().splitlines()
HEADERS = {'Authorization': f"Bearer {tokens[worker_id]}"}

# ----- Signal Handler -----
def handle_exit(signum, frame):
    """Reset user status to pending on exit."""
    global current_processing_user
    if current_processing_user:
        uid = current_processing_user['uid']
        print(f"Interrupted: resetting {uid} to pending...")
        try:
            info_collection.update_one(
                {'uid': uid},
                {'$set': {'processing_status': 'pending'}}
            )
        except Exception as e:
            print(f"Error resetting status: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ----- Error Logging -----
def save_error_log(data_name: str, object_name: str,
                   content: str, res_code='None', error_message=None):
    """Insert an error record into the local error_log collection."""
    local_error_collection.insert_one({
        'loadtime': datetime.now(),
        'data_name': data_name,
        'object': object_name,
        'content': content,
        'response_code': res_code,
        'error_message': str(error_message) if error_message else None
    })

# ----- Fetch Local Account Info -----
def get_local_id(account_info: dict):
    """
    Lookup the user on their local instance to get the true account info.
    Returns JSON dict on success, or False on failure.
    """
    instance = account_info['local_instance_name']
    lookup_url = f"https://{instance}/api/v1/accounts/lookup"
    # Extract username from acct field (username@instance)
    match = re.match(r'^(?P<username>\w+)@', account_info['acct'])
    if not match:
        return False
    username = match.group('username')
    params = {'acct': username}
    retries = 0

    while True:
        try:
            resp = requests.get(lookup_url, headers=HEADERS,
                                params=params, timeout=5)
            status = resp.status_code
            if status == 200:
                judge_sleep(resp.headers, instance,
                            limit_dict, limit_set)
                return resp.json()
            elif status in (429, 503):  # Rate limited or service unavailable
                retries += 1
                time.sleep(2)
                if retries > 3:
                    reset_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                    limit_dict[instance] = reset_time.isoformat().replace('+00:00','Z')
                    limit_set.add(instance)
                    save_error_log('local_id', account_info['uid'],
                                   'Rate limit', res_code=status)
                    return False
                continue
            else:
                save_error_log('local_id', account_info['uid'],
                               'Lookup error', res_code=status)
                return False

        except requests.exceptions.Timeout:
            retries += 1
            time.sleep(0.05)
            if retries > 1:
                reset_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                limit_dict[instance] = reset_time.isoformat().replace('+00:00','Z')
                limit_set.add(instance)
                save_error_log('local_id', account_info['uid'],
                               'Timeout')
                return False
            continue

        except Exception as e:
            save_error_log('local_id', account_info['uid'],
                           'Exception', error_message=e)
            return False

# ----- Fetch Next User to Process -----
def fetch_users_id(social_limit: bool):
    """
    Retrieve the next non-local user pending resolution, skipping rate-limited instances.
    """
    query = {
        'islocal': False,
        'processing_status': 'pending'
    }
    if social_limit:
        query['local_instance_name'] = {'$ne': 'mastodon.social'}
    else:
        query['local_instance_name'] = {'$nin': list(limit_set)}

    candidate = info_collection.find_one_and_update(
        query,
        {'$set': {'processing_status': 'read'}},
        sort=[('bfs_layer', 1)],
    )
    return candidate

# ----- Queue Resolved User -----
def save_user_bfs(account: dict, instance: str, layer: int):
    """Enqueue resolved local user for BFS crawling."""
    rename_key(account, 'id', 'user_id')
    account['acct'] = account_info['acct']
    account['uid'] = f"{account['username']}@{instance}#{account['user_id']}"
    account['bfs_layer'] = layer
    account['processing_status'] = 'pending'
    try:
        users_collection.insert_one(account)
    except DuplicateKeyError:
        pass

# ----- Main Task Processor -----
def process_task(info: dict):
    """
    For a given non-local user, fetch and save their true local ID.
    """
    global current_processing_user
    current_processing_user = info
    now = datetime.now()

    result = get_local_id(info)
    if result:
        # Merge and update account info with resolved data
        save_accountinfo(now, result,
                        info['bfs_layer'],
                        info['local_instance_name'])
    # Mark as completed regardless of success to avoid infinite loops
    info_collection.update_one(
        {'uid': info['uid']},
        {'$set': {'processing_status': 'completed'}}
    )
    current_processing_user = None

# ----- Main Loop -----
if __name__ == '__main__':
    while True:
        try:
            judge_api_islimit(limit_dict, limit_set)
            task = fetch_users_id('mastodon.social' in limit_set)
            if task:
                process_task(task)
            else:
                time.sleep(60)
        except Exception as err:
            print(f"Runtime error: {err}")
            if current_processing_user:
                info_collection.update_one(
                    {'uid': current_processing_user['uid']},
                    {'$set': {'processing_status': 'pending'}}
                )
                current_processing_user = None
            time.sleep(5)
