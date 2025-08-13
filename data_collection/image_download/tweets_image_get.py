#!/usr/bin/env python3
"""
Concurrent Image Downloader

Reads user posts mapping from JSON and account list from CSV, then downloads
all images concurrently, processes formats, saves as JPEG, and records status.
"""

import os
import io
import json
import csv
import logging
import requests
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----- Configuration -----
USER_POSTS_JSON = '../dataset/user_posts.json'
ACCOUNTS_CSV = '../dataset/accounts_labeled.csv'
OUTPUT_FOLDER = '../dataset/downloaded_images'
STATUS_JSON = '../dataset/download_status.json'
MAX_WORKERS = 20

# ----- Logging Setup -----
log_file = 'image_download.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ----- Ensure Output Directory -----
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----- Build Account Index -----
acct_to_idx = {}
with open(ACCOUNTS_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        acct_to_idx[row['acct']] = idx

# ----- Load User Posts -----
with open(USER_POSTS_JSON, 'r', encoding='utf-8') as f:
    user_posts = json.load(f)

# ----- Prepare Download Tasks -----
tasks = []
for acct, posts in user_posts.items():
    if acct not in acct_to_idx:
        continue
    user_idx = acct_to_idx[acct]
    for post in posts:
        post_idx = post['index']
        for url_idx, url in enumerate(post['image_urls']):
            tasks.append((user_idx, post_idx, url_idx, url))

# ----- Download Function -----
def download_and_save(task):
    """
    Download an image URL, convert to JPEG, and save to disk.
    Returns metadata tuple for status tracking.
    """
    user_idx, post_idx, url_idx, url = task
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_stream = io.BytesIO(resp.content)
        img = Image.open(img_stream)
        # Convert modes P or RGBA to RGB
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        save_name = f"{user_idx}_{post_idx}_{url_idx}.jpg"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        img.save(save_path, 'JPEG')
        logging.info(f"Saved {save_path} from {url}")
        return user_idx, post_idx, url_idx, url, True, None
    except requests.RequestException as e:
        logging.error(f"Network error for {url}: {e}")
        return user_idx, post_idx, url_idx, url, False, str(e)
    except Image.UnidentifiedImageError:
        logging.error(f"Invalid image format at {url}")
        return user_idx, post_idx, url_idx, url, False, 'Invalid format'
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        return user_idx, post_idx, url_idx, url, False, str(e)

# ----- Execute Downloads -----
status = defaultdict(lambda: defaultdict(list))
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = [pool.submit(download_and_save, t) for t in tasks]
    for fut in as_completed(futures):
        u, p, i, url, ok, err = fut.result()
        status[u][p].append({
            'url_idx': i,
            'url': url,
            'success': ok,
            'error': err
        })

# ----- Save Status Report -----
with open(STATUS_JSON, 'w', encoding='utf-8') as f:
    json.dump(status, f, ensure_ascii=False, indent=2)

# ----- Summary Logging -----
total = len(tasks)
saved = sum(1 for name in os.listdir(OUTPUT_FOLDER) if name.endswith('.jpg'))
logging.info(f"Total URLs: {total}, Images saved: {saved}")
logging.info(f"Images in: {OUTPUT_FOLDER}, Status in: {STATUS_JSON}")
