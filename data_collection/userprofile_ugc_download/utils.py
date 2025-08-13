#!/usr/bin/env python3
"""
Utility functions for MongoDB indexing, API rate-limit handling,
and dictionary key manipulation.
"""

import time
import re
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError, OperationFailure

# ----- MongoDB Index Helpers -----

def create_unique_index(collection, field_name):
    """
    Ensure a unique index exists on the specified field in the collection.
    If the index does not exist, create it; handle duplicate key errors gracefully.
    """
    indexes = collection.index_information()
    if field_name not in indexes:
        try:
            collection.create_index([(field_name, 1)], unique=True)
            print(f"Created unique index on '{field_name}' for collection '{collection.name}'.")
        except DuplicateKeyError:
            print(f"DuplicateKeyError: existing documents violate unique constraint on '{field_name}'.")
        except Exception as e:
            print(f"Error creating unique index on '{field_name}': {e}")
    else:
        print(f"Unique index on '{field_name}' already exists for collection '{collection.name}'.")


def create_compound_index(collection, *fields):
    """
    Create a compound index on the specified fields in the collection.
    Checks for existing index before creation and reports status.
    """
    index_name = "_".join([f"{f}_1" for f in fields])
    existing = collection.index_information()

    if index_name in existing:
        print(f"Compound index '{index_name}' already exists in collection '{collection.name}'.")
        return

    try:
        specs = [(field, 1) for field in fields]
        collection.create_index(specs)
        print(f"Created compound index on {fields} for collection '{collection.name}'.")
    except OperationFailure as e:
        print(f"OperationFailure creating compound index on {fields}: {e}")
    except Exception as e:
        print(f"Error creating compound index on {fields}: {e}")

# ----- API Rate-Limit Helpers -----

def judge_sleep(headers, instance, limit_dict, limit_set):
    """
    If the API rate limit is exceeded or nearing exhaustion,
    calculate reset time, sleep until reset, and update limit tracking.

    Returns:
        False if the function sleeps, True otherwise.
    """
    low_headers = {k.lower(): v for k, v in headers.items()}
    remaining = int(low_headers.get('x-ratelimit-remaining', 1))
    reset_str = low_headers.get('x-ratelimit-reset')

    # If no calls remain, sleep until reset
    if remaining <= 0 and reset_str:
        reset_time = _parse_iso_timestamp(reset_str)
        now = datetime.now(timezone.utc)
        sleep_seconds = (reset_time - now).total_seconds()
        if sleep_seconds > 0:
            print(f"Rate limit reached for '{instance}', sleeping until {reset_time}...")
            time.sleep(sleep_seconds)
            return False

    # If calls are low (<100), mark instance as limited
    if remaining <= 100 and reset_str:
        reset_time = _parse_iso_timestamp(reset_str)
        now = datetime.now(timezone.utc)
        if reset_time > now:
            limit_dict[instance] = reset_time.isoformat()
            limit_set.add(instance)
            print(f"Instance '{instance}' added to limit set until {reset_time}.")
            return True

    return True


def judge_api_islimit(limit_dict, limit_set):
    """
    Remove instances from the limit set if their reset time has passed.
    """
    now = datetime.now(timezone.utc)
    expired = [inst for inst, ts in limit_dict.items()
               if datetime.fromisoformat(ts) <= now]
    for inst in expired:
        limit_set.discard(inst)
        del limit_dict[inst]
        print(f"Instance '{inst}' removed from limit set.")

# ----- Dictionary Utilities -----

def rename_key(data: dict, old_key: str, new_key: str) -> dict:
    """
    Rename a key in the dictionary if it exists, returning the updated dict.
    """
    if old_key in data:
        data[new_key] = data.pop(old_key)
    return data

# ----- Internal Helpers -----

def _parse_iso_timestamp(ts: str) -> datetime:
    """
    Parse an ISO-8601 timestamp string, handling 'Z' suffix.
    Returns a datetime with UTC timezone.
    """
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    return datetime.fromisoformat(ts).astimezone(timezone.utc)
