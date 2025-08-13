# Image Classification with Qwen 2.5 VL-32B Instruct

Automated large-scale image classification using Qwen/Qwen2.5-VL-32B-Instruct vision model. Processes images in parallel and outputs results to CSV files.

## Features

- **Parallel batch processing** using configurable thread pool (32 workers)
- **Checkpointing**: saves progress every 100 images or 300 seconds
- **Detailed logging** of successes, failures, and token usage
- **26 Categories**: Product, Food, Finance, Architecture, Sport, etc.

## Prerequisites

- Python 3.8+
- OpenAI Python client: `pip install openai`
- Valid API key for Qwen model access

## Setup

1. **Set API Key**:
   ```bash
   export DASHSCOPE_API_KEY="your-api-key-here"
   ```

3. **Change BASE_URL**
    - Update the `BASE_URL` variable in the script.

4. **Prepare Image Directory**:
   - Place images in `../dataset/autodl-tmp/60k_jpg/`
   - Supported formats: `.jpg`, `.jpeg`, `.png`

5. **Configure Script** (optional):
   - Edit `IMAGE_DIR`, `OUTPUT_CSV`, `FAILED_CSV` paths
   - Adjust `MAX_WORKERS` and save intervals

## Usage

```bash
python image_classification_60k.py
```

## Output

- **Success**: `../dataset/60k_image_classification.csv`
  - Columns: image_name, category, prompt_tokens, completion_tokens, total_tokens
- **Failures**: `../dataset/60k_failed_image_classification.csv`
  - Columns: image_name, failure_reason
- **Logs**: `classification.log`

## Categories

The model classifies images into 26 categories:
1. Product, 2. Food & Beverage, 3. Finance, 4. Architecture, 5. Sport, 6. Transport, 7. Art, 8. Urban, 9. Nature, 10. People, 11. Event, 12. History & Culture, 13. Animation & Comics, 14. Science & Technology, 15. Statistical Data, 16. Poster, 17. Interior, 18. Game, 19. Snapshot, 20. Education, 21. News & Newspaper, 22. Tourism, 23. Advertisement, 24. Meme, 25. UI, 26. Others

