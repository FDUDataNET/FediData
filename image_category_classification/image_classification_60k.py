#!/usr/bin/env python3
"""
Image classification script: classifies images using OpenAI API and writes results to CSV.
"""

import os
import time
import base64
import csv
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# ---------- Configuration ----------
API_KEY_ENV = "DASHSCOPE_API_KEY"
BASE_URL = "https://ai.forestsx.top/v1"
MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
IMAGE_DIR = "../dataset/autodl-tmp/60k_jpg"
OUTPUT_CSV = "../dataset/60k_image_classification.csv"
FAILED_CSV = "../dataset/60k_failed_image_classification.csv"

SAVE_INTERVAL = 100        # Save progress every N images
TIME_INTERVAL = 300        # Save progress every N seconds
TOTAL_IMAGES = 765_019     # Total number of images to process
MAX_WORKERS = 32           # Number of concurrent threads

# ---------- Logging Setup ----------
logging.basicConfig(
    filename="classification.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- Initialize OpenAI Client ----------
api_key = os.getenv(API_KEY_ENV)
if not api_key:
    raise EnvironmentError(f"Environment variable {API_KEY_ENV} not set")
client = OpenAI(api_key=api_key, base_url=BASE_URL)

# ---------- Prepare CSV Files ----------
def initialize_csv_files():
    """Create CSV files with headers for successful and failed classifications."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(
            out_file,
            fieldnames=[
                "image_name",
                "category",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ],
        )
        writer.writeheader()

    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as fail_file:
        writer = csv.DictWriter(
            fail_file, fieldnames=["image_name", "failure_reason"]
        )
        writer.writeheader()

# ---------- Message Preparation ----------
def prepare_messages(image_path: str) -> list:
    """
    Read an image file and encode it into data URI for classification.
    Returns the list of messages to send to the OpenAI API.
    """
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    data_uri = (
        "data:image/jpeg;base64,"
        + base64.b64encode(img_bytes).decode("utf-8")
    )

    system_msg = {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are an image classification assistant."}
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {
                "type": "text",
                "text": (
                    "Please perform an image classification task and assign the given "
                    "image to one of the following categories:\n\n"
                    "1. Product\n2. Food & Beverage\n3. Finance\n4. Architecture\n"
                    "5. Sport\n6. Transport\n7. Art\n8. Urban\n9. Nature\n10. People\n"
                    "11. Event\n12. History & Culture\n13. Animation & Comics\n"
                    "14. Science & Technology\n15. Statistical Data\n16. Poster\n"
                    "17. Interior\n18. Game\n19. Snapshot\n20. Education\n"
                    "21. News & Newspaper\n22. Tourism\n23. Advertisement\n"
                    "24. Meme\n25. UI\n26. Others\n\n"
                    "Note:\n"
                    "1. Avoid assigning an image to multiple categories.\n"
                    "2. Output only the category name without any additional response."
                ),
            },
        ],
    }
    return [system_msg, user_msg]

# ---------- Image Processing ----------
processed_count = 0
last_save_time = time.time()

def process_image(filename: str):
    """
    Classify a single image and append the result to the CSV files.
    Logs successes and failures.
    """
    global processed_count, last_save_time

    image_path = os.path.join(IMAGE_DIR, filename)
    try:
        messages = prepare_messages(image_path)
        response = client.chat.completions.create(
            model=MODEL, messages=messages
        )
        category = response.choices[0].message.content.strip()

        if not category or category.lower() in ("error", "n/a"):
            raise ValueError("No valid category returned")

        usage = response.usage
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(
                out_file,
                fieldnames=[
                    "image_name",
                    "category",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ],
            )
            writer.writerow(
                {
                    "image_name": filename,
                    "category": category,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )
        print(
            f"{filename} -> {category} "
            f"[Processed: {processed_count + 1}/{TOTAL_IMAGES}, "
            f"Remaining: {TOTAL_IMAGES - processed_count - 1}]"
        )
        logging.info(f"{filename} -> {category}")

    except Exception as e:
        # Extract relevant message
        message = str(e)
        if "'message'" in message:
            try:
                start = message.find("'message': '") + len("'message': '")
                end = message.find("'", start)
                message = message[start:end]
            except Exception:
                pass

        print(
            f"Error processing {filename}: {message} "
            f"[Processed: {processed_count + 1}/{TOTAL_IMAGES}, "
            f"Remaining: {TOTAL_IMAGES - processed_count - 1}]"
        )
        logging.error(f"{filename} -> {message}")
        with open(FAILED_CSV, "a", newline="", encoding="utf-8") as fail_file:
            writer = csv.DictWriter(
                fail_file, fieldnames=["image_name", "failure_reason"]
            )
            writer.writerow(
                {"image_name": filename, "failure_reason": message}
            )

    processed_count += 1

    # Periodic progress saving
    current_time = time.time()
    if (
        processed_count % SAVE_INTERVAL == 0
        or (current_time - last_save_time) >= TIME_INTERVAL
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Saved progress at {timestamp} (processed {processed_count}/{TOTAL_IMAGES})")
        logging.info(f"Saved progress at {timestamp} (processed {processed_count}/{TOTAL_IMAGES})")
        last_save_time = current_time

def main():
    """Main entry point: initialize CSVs, spawn threads to process images."""
    initialize_csv_files()
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_image, image_files)

    # Final save/log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Final save at {timestamp} (total processed: {processed_count}/{TOTAL_IMAGES})")
    logging.info(f"Final save at {timestamp} (total processed: {processed_count}/{TOTAL_IMAGES})")

if __name__ == "__main__":
    main()
