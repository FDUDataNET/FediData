import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

def read_jsonl_data(file_path, instance_name):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            content = obj['response']['body']['choices'][0]['message']['content']
            records.append({'classification': content, 'instance': instance_name})
    return records

mastodon_records = read_jsonl_data('./dataset/batch_output/mastodon.social_output.jsonl', 'mastodon.social')
chaos_records = read_jsonl_data('./dataset/batch_output/chaos.social_output.jsonl', 'chaos.social')

all_records = mastodon_records + chaos_records
dataframe = pd.DataFrame(all_records)

dataframe[['topic', 'emotion']] = dataframe['classification'].str.split(' - ', expand=True)

valid_topics = [
    'arts & culture', 'business & finance', 'careers', 'entertainment', 
    'fashion & beauty', 'food', 'gaming', 'hobbies & interests', 
    'movies & TV', 'music', 'news', 'outdoors', 'science', 
    'sports', 'technology', 'travel'
]

dataframe.loc[~dataframe['topic'].isin(valid_topics), 'topic'] = 'none'

dataframe_filtered = dataframe[dataframe['topic'] != 'none']

topic_colors = [
    '#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc',
    '#e5d8bd', '#fddaec', '#f2f2f2', '#b3e2cd', '#fdcdac', '#cbd5e8',
    '#f4cae4', '#e6f5c9', '#fff2ae', '#f1e2cc', '#cccccc', '#ffcccc',
    '#ccffcc', '#ccccff'
]

all_topics = dataframe_filtered['topic'].value_counts().index.tolist()
topic_color_map = {topic: topic_colors[i % len(topic_colors)] for i, topic in enumerate(all_topics)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

instances = ['mastodon.social', 'chaos.social']
axes = [ax1, ax2]

for i, (instance, ax) in enumerate(zip(instances, axes)):
    instance_data = dataframe_filtered[dataframe_filtered['instance'] == instance]
    topic_counts = instance_data['topic'].value_counts()
    
    colors = [topic_color_map[topic] for topic in topic_counts.index]
    
    wedges, _ = ax.pie(
        topic_counts.values,
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    ax.set_title(f'{instance}', fontsize=14, fontweight='bold')

legend_elements = [plt.Rectangle((0,0),1,1, facecolor=topic_color_map[topic], edgecolor='white') 
                   for topic in all_topics]

ax2.legend(
    legend_elements,
    all_topics,
    title='Topics',
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    fontsize=10,
    title_fontsize=12,
    frameon=False
)

plt.tight_layout()
plt.savefig("topic_distribution_comparison.pdf")


news_data = dataframe_filtered[dataframe_filtered['topic'] == 'news']
instances = ['mastodon.social', 'chaos.social']

counts = (
    news_data
    .groupby(['instance','emotion'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=['neutral','negative','positive'])
)
percent = counts.div(counts.sum(axis=1), axis=0) * 100

emotion_colors = {
    'neutral':  '#b3cde3',
    'negative': '#ccebc5',
    'positive': '#fbb4ae'
}
hatches = {
    'neutral':  '////',
    'negative': 'xxxx',
    'positive': '....'
}

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(instances))
width = 0.2

for i, emo in enumerate(['neutral', 'negative', 'positive']):
    offsets = x + (i - 1) * width
    vals = percent[emo].values
    bars = ax.bar(
        offsets, vals, width,
        label=emo.capitalize(),
        color=emotion_colors[emo],
        hatch=hatches[emo],
        edgecolor='white',
        linewidth=1.2
    )
    
    for j, v in enumerate(vals):
        ax.text(
            offsets[j], v + 1,
            f'{v:.1f}%',
            ha='center', va='bottom',
            fontsize=10
        )

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False,
    fontsize=12,
    title_fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(instances, fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_title('Emotion of "news"', fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

plt.savefig("emotion_analysis.pdf")
