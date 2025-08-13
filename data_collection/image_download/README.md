# Concurrent Image Downloader

Downloads images from Mastodon posts concurrently, converts them to JPEG format, and tracks download status. Processes image URLs extracted from user-generated content (UGC) data.

## Prerequisites

### Dependencies
```bash
pip install requests pillow
```

### Required Input Files

1. **user_posts.json** (from UGC extraction)
   - Contains mapping of user accounts to their posts with image URLs
   - **Note**: This file is extracted from UGC collection data. Code for this extraction will be updated in future versions.
   - Format:
   ```json
   {
     "user@instance.social": [
       {
         "index": 0,
         "image_urls": ["https://...", "https://..."]
       }
     ]
   }
   ```

2. **accounts_labeled.csv**
   - CSV file with user account information
   - Must contain 'acct' column with user identifiers
   - Used for mapping accounts to indices

## Usage

### Step 1: Prepare Input Files

Ensure you have the required input files in the correct locations:
```bash
../dataset/user_posts.json      # From UGC extraction
../dataset/accounts_labeled.csv # User account mapping
```

### Step 2: Run Image Downloader
```bash
python tweets_image_get.py
```

### Configuration Options

Edit the script's configuration section as needed:
```python
USER_POSTS_JSON = '../dataset/user_posts.json'
ACCOUNTS_CSV = '../dataset/accounts_labeled.csv'
OUTPUT_FOLDER = '../dataset/downloaded_images'
STATUS_JSON = '../dataset/download_status.json'
MAX_WORKERS = 20  # Adjust based on your system and network
```

## Output

### Downloaded Images
- **Location**: `../dataset/downloaded_images/`
- **Format**: JPEG files
- **Naming**: `{userIndex}_{postIndex}_{urlIndex}.jpg`
- **Processing**: Automatic conversion from P/RGBA to RGB

### Status Tracking
- **File**: `../dataset/download_status.json`
- **Content**: Nested JSON with success/failure status for each image
- **Structure**:
```json
{
  "userIndex": {
    "postIndex": [
      {
        "url_idx": 0,
        "url": "https://...",
        "success": true,
        "error": null
      }
    ]
  }
}
```

### Logs
- **File**: `image_download.log`
- **Content**: Detailed download progress, errors, and summary
- **Console**: Real-time progress output


