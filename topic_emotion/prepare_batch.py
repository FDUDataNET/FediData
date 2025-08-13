import json
import os

with open("./dataset/ugc_all_instance.json") as f:
    data = json.load(f)

base_prompt = (
    "please classify this post into one of the topics "
    "arts & culture, business & finance, careers, entertainment, "
    "fashion & beauty, food, gaming, hobbies & interests, movies & tv, "
    "music, news, outdoors, science, sports, technology, travel "
    "and an emotion (positive, negative, neutral). "
    "format as 'topic - emotion'. if unsure topic, use 'none'. "
    "lowercase and under 100 words.\n\npost: {tweet}"
)

if not os.path.exists("./dataset/batch_files"):
    os.makedirs("./dataset/batch_files")

print(f"{len(data.keys())} instances")

for instance_name, tweets in data.items():
    
    batch_requests = []
    request_mapping = {}
    request_id = 1
    
    for i, tweet in enumerate(tweets):
        prompt = base_prompt.format(tweet=tweet)
        
        batch_request = {
            "custom_id": f"request-{request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": "you are a helpful classifier."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10
            }
        }
        
        batch_requests.append(batch_request)
        
        request_mapping[f"request-{request_id}"] = {
            "instance": instance_name,
            "tweet_index": i,
            "tweet": tweet
        }
        
        request_id += 1
    
    batch_filename = f"./dataset/batch_files/{instance_name}_batch_requests.jsonl"
    with open(batch_filename, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    mapping_filename = f"./dataset/batch_files/{instance_name}_request_mapping.json"
    with open(mapping_filename, "w", encoding="utf-8") as f:
        json.dump(request_mapping, f, ensure_ascii=False, indent=2)
    

print(f"\nComplete! Created batch files for {len(data.keys())} instances")