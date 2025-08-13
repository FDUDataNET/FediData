#!/usr/bin/env python3
"""
Fetch Mastodon Instances Info

Retrieves metadata for all Mastodon instances via the instances.social API,
stores results in MongoDB, and saves instance names to a local file.
"""

import json
import requests
from datetime import datetime
from pymongo import MongoClient, errors
from utils import create_unique_index

# ----- Configuration -----
API_TOKEN = (
    
)
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}
API_URL = 'https://instances.social/api/1.0/instances/list'
MONGO_URI = (
  
)
DB_NAME = 'mastodon'
COLLECTION_NAME = 'instances'
OUTPUT_FILE = 'instances_list.txt'

# ----- Fetch Data -----
def fetch_instances():
    """
    Send request to instances.social API and return parsed JSON data.
    """
    params = {'count': 0}
    response = requests.get(API_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

# ----- Save to MongoDB and File -----
def save_to_mongo(db, instances, loadtime):
    """
    Insert instance documents into MongoDB collection with unique index on 'name'.
    """
    coll = db[COLLECTION_NAME]
    create_unique_index(coll, 'name')

    inserted = 0
    for instance in instances:
        doc = instance.copy()
        doc['loadtime'] = loadtime
        doc['processing_status'] = 'pending'
        doc['statuses'] = int(doc.get('statuses', 0))
        try:
            coll.insert_one(doc)
            inserted += 1
        except errors.DuplicateKeyError:
            continue
    print(f"Inserted {inserted} new instances into MongoDB.")


def save_names_file(instances, filename):
    """
    Write each instance 'name' to a newline-delimited text file.
    """
    names = [inst['name'] for inst in instances if 'name' in inst]
    with open(filename, 'w') as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Saved {len(names)} instance names to {filename}.")

# ----- Main Execution -----
def main():
    # Fetch data
    data = fetch_instances()
    instances = data.get('instances', [])
    now = datetime.now()

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Save
    save_to_mongo(db, instances, now)
    save_names_file(instances, OUTPUT_FILE)

    client.close()

if __name__ == '__main__':
    main()
