# Topic & Emotion Analysis
## data prepare
Ensure the UGC JSON file named as `ugc_all_instance.json`, the data form should be:

```json
{
  "instance1.social": ["post1", "post2", "post3", ...],
  "instance2.com": ["post1", "post2", "post3", ...],
  ...
}
```

## run prepare_batch.py
Generate classification prompts and requests. This will generate request files for each instance inside `./dataset/batch_files/*_batch_requests.jsonl`.

**Submit the `*_batch_requests.jsonl` files to OpenAI’s Batch API. Once completed, save the results to: `./dataset/batch_output/*_output.jsonl`.**

## run visualize.py
Analyze and visualize the topic distribution and emotional sentiment of posts from two instances (mastodon.social and chaos.social) using pie charts and a comparative bar chart.
