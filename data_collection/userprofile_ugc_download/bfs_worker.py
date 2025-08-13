#!/usr/bin/env python3
"""
Mastodon BFS Crawler Worker

Crawls user relationships (followers and followings) from Mastodon instances,
resolves local IDs, and stores data in MongoDB.
"""

import re
import sys
import json
import time
import signal
import argparse
import requests
from datetime import datetime, timezone, timedelta

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, OperationFailure
from utils import (
    create_unique_index,
    judge_sleep,
    judge_api_islimit,
    rename_key
)

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(description='BFS worker for Mastodon user relationships.')
parser.add_argument('--id', type=int, required=True, help='Worker ID (index into token list)')
parser.add_argument('--priority', type=int, default=0, help='Worker priority')
args = parser.parse_args()
worker_id = args.id
priority = args.priority

# ----- Load Instance List -----
instances_list = set()
with open('instances_list.txt', 'r') as file:
    for line in file:
        instances_list.add(line.strip())  # remove newline

# ----- Load API Token -----
limit_dict = {}
limit_set = set()
with open('token_list.txt', 'r') as f:
    tokens = f.read().splitlines()
HEADERS = {'Authorization': f'Bearer {tokens[worker_id]}'}

# ----- MongoDB Setup -----
# Remote MongoDB
REMOTE_URI = ''
remote_client = MongoClient(REMOTE_URI)
remote_db = remote_client['mastodon']
users_collection = remote_db['users_bfs']
info_collection = remote_db['accounts_info']

# Ensure unique indexes
create_unique_index(users_collection, 'acct')
create_unique_index(info_collection, 'uid')

# Local MongoDB
local_client = MongoClient('mongodb://localhost:27018/')
local_db = local_client['mastodon']
local_relation_collection = local_db['relation']
local_error_collection = local_db['error_log']

current_processing_user = None

# ----- Signal Handlers -----
def handle_exit(signal_num, frame):
    """Cleanup and mark current user as pending on exit."""
    global current_processing_user
    if current_processing_user:
        uid = current_processing_user['uid']
        print(f"Interrupted: resetting {uid} status to pending...")
        users_collection.update_one(
            {'uid': uid},
            {'$set': {'bfs_status': 'pending'}}
        )
    print("Cleanup done. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ----- Data Fetching Functions -----
def get_data(data_name: str, user_id: str, instance: str, uid: str):
    """
    Fetch 'followers' or 'following' for a user from a Mastodon instance.
    Returns a list of account dicts or False on failure.
    """
    url = f"https://{instance}/api/v1/accounts/{user_id}/{data_name}"
    all_data = []
    max_id = None
    retries = 0

    while True:
        params = {'limit': 80}
        if max_id:
            params['max_id'] = max_id
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=5)
            status = resp.status_code

            if status == 200:
                judge_sleep(resp.headers, instance, limit_dict, limit_set)
                page = resp.json()
                all_data.extend(page)
                # Check pagination
                if 'Link' not in resp.headers or len(page) < 80:
                    break
                match = re.search(r'max_id=(\d+)', resp.headers['Link'])
                max_id = match.group(1) if match else None
                time.sleep(0.02)

            elif status in (429, 503):
                retries += 1
                time.sleep(2)
                if retries > 3:
                    # Rate limit reached, add to limit set
                    until = (datetime.now(timezone.utc) + timedelta(minutes=5))
                    iso_until = until.isoformat().replace('+00:00', 'Z')
                    limit_dict[instance] = iso_until
                    limit_set.add(instance)
                    save_error_log(data_name, uid, 'Rate limit', res_code=status)
                    return False
                continue

            else:
                save_error_log(data_name, uid, 'HTTP error', res_code=status)
                return False

        except requests.exceptions.Timeout:
            retries += 1
            if retries > 1:
                until = (datetime.now(timezone.utc) + timedelta(minutes=5))
                iso_until = until.isoformat().replace('+00:00', 'Z')
                limit_dict[instance] = iso_until
                limit_set.add(instance)
                save_error_log(data_name, uid, 'Timeout')
                return False
            continue

        except Exception as e:
            save_error_log(data_name, uid, 'Exception', error_message=e)
            return False

    return all_data

# ----- Database Save Helpers -----
def save_error_log(data_name: str, object_name: str, content: str,
                   res_code: str = 'None', error_message=None):
    """Insert an error record into local error_log collection."""
    local_error_collection.insert_one({
        'loadtime': datetime.now(),
        'data_name': data_name,
        'object': object_name,
        'content': content,
        'response_code': res_code,
        'error_message': str(error_message)
    })


def save_relation(relation: dict):
    """Insert a relationship document into local relation collection."""
    try:
        local_relation_collection.insert_one(relation)
    except DuplicateKeyError:
        pass


def save_user_bfs(account: dict, instance: str, layer: int):
    """Queue a user for BFS crawling."""
    doc = {
        'acct': f"{account['username']}@{instance}",
        'uid': account['uid'],
        'user_id': account['user_id'],
        'instance_name': instance,
        'url': account['url'],
        'statuses_count': int(account['statuses_count']),
        'followers_count': int(account['followers_count']),
        'following_count': int(account['following_count']),
        'bfs_status': 'pending',
        'ugc_status': 'pending',
        'bfs_layer': layer
    }
    if doc['followers_count'] <= 0 and doc['following_count'] <= 0:
        doc['bfs_status'] = 'completed'
    if doc['statuses_count'] <= 0:
        doc['ugc_status'] = 'completed'
    try:
        users_collection.insert_one(doc)
    except DuplicateKeyError:
        pass


def save_accountinfo(current_time: datetime, account: dict,
                     layer: int, remote_instance: str):
    """Store fetched account info and enqueue if local or remote BFS required."""
    rename_key(account, 'id', 'user_id')
    account['loadtime'] = current_time
    # Parse URL to extract local instance and username
    m = re.search(r'https?://([^/]+)/@([^/]+)', account['url'])
    if not m:
        account['isin_fediverselist'] = False
    else:
        local_inst, username = m.groups()
        account['local_instance_name'] = local_inst
        account['acct'] = f"{username}@{local_inst}"
        account['uid'] = f"{username}@{remote_instance}#{account['user_id']}"
        account['isin_fediverselist'] = (local_inst in instances_list)
        account['islocal'] = (remote_instance == local_inst)
        account['bfs_layer'] = layer
        account['processing_status'] = 'completed' if account['islocal'] else 'pending'
        if not account['isin_fediverselist']:
            account['processing_status'] = 'skipped'
        if account['islocal']:
            save_user_bfs(account, local_inst, layer)
    try:
        info_collection.insert_one(account)
    except DuplicateKeyError:
        pass

# ----- Main Processing -----
def process_task(user_info: dict):
    """Handle one BFS user: fetch followers and following, save relations."""
    global current_processing_user
    current_processing_user = user_info

    uid = user_info['uid']
    instance = user_info['instance_name']
    layer = user_info['bfs_layer']

    following = get_data('following', user_info['user_id'], instance, uid)
    followers = get_data('followers', user_info['user_id'], instance, uid)
    now = datetime.now()

    if following is False or followers is False:
        users_collection.update_one({'uid': uid}, {'$set': {'bfs_status': 'pending'}})
        return

    # Save account info and collect account handles
    for item in followers + following:
        save_accountinfo(now, item, layer + 1, instance)

    relation_doc = {
        'acct': user_info['acct'],
        'uid': uid,
        'followers': [f"{re.search(r'https?://([^/]+)/@([^/]+)', x['url']).group(2)}@{re.search(r'https?://([^/]+)/', x['url']).group(1)}" for x in followers],
        'following': [f"{re.search(r'https?://([^/]+)/@([^/]+)', x['url']).group(2)}@{re.search(r'https?://([^/]+)/', x['url']).group(1)}" for x in following],
        'loadtime': now
    }
    save_relation(relation_doc)
    users_collection.update_one({'uid': uid}, {'$set': {'bfs_status': 'completed'}})
    current_processing_user = None


def fetch_users_id(social_limited: bool):
    """Fetch one pending user for BFS, avoiding rate-limited instances."""
    query = {'bfs_status': 'pending'}
    if social_limited:
        query['instance_name'] = {'$ne': 'mastodon.social'}
    else:
        query['instance_name'] = {'$nin': list(limit_set)}

    candidate = users_collection.find_one_and_update(
        query,
        {'$set': {'bfs_status': 'read'}},
        sort=[('bfs_layer', 1)]
    )
    return candidate


def main():
    """Main loop: continuously fetch and process BFS tasks."""
    while True:
        try:
            judge_api_islimit(limit_dict, limit_set)
            user = fetch_users_id('mastodon.social' in limit_set)
            if user:
                process_task(user)
            else:
                time.sleep(60)
        except Exception as e:
            print(f"Runtime error: {e}")
            if current_processing_user:
                users_collection.update_one(
                    {'uid': current_processing_user['uid']},
                    {'$set': {'bfs_status': 'pending'}}
                )
                current_processing_user = None
            time.sleep(5)

if __name__ == '__main__':
    main()
