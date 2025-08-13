#!/usr/bin/env python3
"""
UGC Worker for Mastodon

Fetches user statuses (UGC) from Mastodon instances and stores them in local MongoDB.
Handles API rate limits, errors, and supports graceful shutdown.
"""

import argparse
import signal
import sys
import time
import re
from datetime import datetime, timedelta, timezone

import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from utils import judge_sleep, judge_api_islimit

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(
    description='UGC worker for Mastodon users.'
)
parser.add_argument(
    '--id', type=int, required=True,
    help='Worker ID (index into token list)'
)
parser.add_argument(
    '--priority', type=int, default=0,
    help='Worker priority for instance selection'
)
args = parser.parse_args()
worker_id = args.id
priority = args.priority

# ----- Load API Token -----
with open('token_list.txt') as f:
    tokens = f.read().splitlines()
HEADERS = {'Authorization': f'Bearer {tokens[worker_id]}'}

# ----- MongoDB Setup -----
# Remote users collection
REMOTE_URI = 'mongodb://USERNAME:PASSWORD@HOST:PORT/'
remote_client = MongoClient(REMOTE_URI)
users_collection = remote_client['mastodon']['users_bfs']

# Local UGC and error log collections
local_client = MongoClient('mongodb://localhost:27018/')
local_db = local_client['mastodon']
ugc_collection = local_db['ugc']
error_collection = local_db['error_log']

# ----- Global State -----
limit_dict = {}
limit_set = set()
current_processing_user = None

# ----- Signal Handler -----
def handle_exit(signum, frame):
    """Reset UGC status of the current user on exit."""
    global current_processing_user
    if current_processing_user:
        uid = current_processing_user['uid']
        print(f"Interrupted: resetting {uid} ugc_status to pending...")
        users_collection.update_one(
            {'uid': uid},
            {'$set': {'ugc_status': 'pending'}}
        )
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ----- Error Logging -----
def save_error_log(data_name: str, object_name: str,
                   content: str, res_code='None', error_message=None):
    """Insert an error record into the local error_log collection."""
    error_collection.insert_one({
        'loadtime': datetime.now(),
        'data_name': data_name,
        'object': object_name,
        'content': content,
        'response_code': res_code,
        'error_message': str(error_message) if error_message else None
    })

# ----- Fetch Account Statuses -----
def get_account_status(user_id: str, instance: str, uid: str):
    """
    Retrieve all statuses for a user from a Mastodon instance.
    Returns a list of status dicts (may be empty).
    """
    base_url = f'https://{instance}/api/v1/accounts/{user_id}/statuses'
    statuses = []
    max_id = None
    retries = 0

    while True:
        params = {
            'limit': 40,
            'exclude_reblogs': False,
            'exclude_replies': False,
            'only_media': False
        }
        if max_id:
            params['max_id'] = max_id

        try:
            resp = requests.get(base_url, headers=HEADERS,
                                params=params, timeout=5)
            status_code = resp.status_code

            if status_code == 200:
                judge_sleep(resp.headers, instance, limit_dict, limit_set)
                page = resp.json()
                statuses.extend(page)
                print(f"Fetched {len(page)} statuses for {uid}")

                headers_lower = {k.lower(): v for k, v in resp.headers.items()}
                if 'link' not in headers_lower or len(page) < 40:
                    break
                match = re.search(r'max_id=(\d+)', headers_lower['link'])
                max_id = match.group(1) if match else None

            elif status_code in (429, 503):
                retries += 1
                time.sleep(2)
                if retries > 3:
                    reset_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                    limit_dict[instance] = reset_time.isoformat()
                    limit_set.add(instance)
                    save_error_log('UGC', uid, 'Rate limit', res_code=status_code)
                    return statuses
                continue

            else:
                save_error_log('UGC', uid, 'HTTP error', res_code=status_code)
                return statuses

        except requests.exceptions.Timeout:
            retries += 1
            time.sleep(0.5)
            if retries > 3:
                reset_time = datetime.now(timezone.utc) + timedelta(minutes=5)
                limit_dict[instance] = reset_time.isoformat()
                limit_set.add(instance)
                save_error_log('UGC', uid, 'Timeout')
                return statuses
            continue

        except Exception as e:
            save_error_log('UGC', uid, 'Exception', error_message=e)
            return statuses

    return statuses

# ----- Save Statuses -----
def save_statuses(user_info: dict, statuses: list):
    """
    Process and insert statuses into the local UGC collection.
    """
    acct = user_info['acct']
    uid = user_info['uid']
    instance = user_info['instance_name']

    docs = []
    for status in statuses:
        sid = f"{instance}#{status['id']}"
        doc = {
            'acct': acct,
            'uid': uid,
            'instance_name': instance,
            'sid': sid,
            'content': status.get('content'),
            'created_at': status.get('created_at'),
            'reblog_status_id': (
                status['reblog']['id'] if status.get('reblog') else None
            )
        }
        docs.append(doc)

    if docs:
        try:
            ugc_collection.insert_many(docs, ordered=False)
            print(f"Saved {len(docs)} statuses for {acct}")
        except DuplicateKeyError:
            print(f"Some statuses for {acct} were already stored.")
        except Exception as e:
            print(f"Error saving statuses for {acct}: {e}")

# ----- Task Processing -----
def process_task(user_info: dict):
    """Fetch and save statuses for one user, then mark UGC completed."""
    user_id = user_info['user_id']
    instance = user_info['instance_name']
    uid = user_info['uid']

    statuses = get_account_status(user_id, instance, uid)
    if statuses:
        save_statuses(user_info, statuses)

    users_collection.update_one(
        {'uid': uid},
        {'$set': {'ugc_status': 'completed'}}
    )
    global current_processing_user
    current_processing_user = None

# ----- Fetch Next User -----
def fetch_next_user(social_limit: bool) -> dict:
    """
    Retrieve the next user pending UGC fetch, skipping limited instances.
    """
    query = {'ugc_status': 'pending'}
    if social_limit:
        query['instance_name'] = {'$ne': 'mastodon.social'}
    else:
        query['instance_name'] = {'$nin': list(limit_set)}

    return users_collection.find_one_and_update(
        query,
        {'$set': {'ugc_status': 'read'}},
        sort=[('bfs_layer', 1)],
    )

# ----- Main Loop -----
def main():
    while True:
        try:
            judge_api_islimit(limit_dict, limit_set)
            user = fetch_next_user('mastodon.social' in limit_set)
            if user:
                global current_processing_user
                current_processing_user = user
                process_task(user)
            else:
                time.sleep(60)
        except Exception as e:
            print(f"Runtime error: {e}")
            if current_processing_user:
                users_collection.update_one(
                    {'uid': current_processing_user['uid']},
                    {'$set': {'ugc_status': 'pending'}}
                )
                current_processing_user = None
            time.sleep(5)

if __name__ == '__main__':
    main()
