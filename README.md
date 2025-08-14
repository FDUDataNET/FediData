# FediData: A Comprehensive Multi-Modal Fediverse Dataset from Mastodon

[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/15621244)

FediData, the first open multi-modal dataset collected from Mastodon, which is dedicated to providing realistic and reliable data support for social behavior modeling, multi-modal learning, and research on user interaction mechanisms.

[📥 Download Dataset](https://zenodo.org/records/15621244)

## Citation
If you use FediData in a scientific publication, we kindly request that you cite the following paper:
```
  @inproceedings{gao2025fedidata,
      title={{FediData: A Comprehensive Multi-Modal Fediverse Dataset from Mastodon}},
      author={Min Gao and Haoran Du and Wen Wen and Qiang Duan and Xin Wang and Yang Chen},
      year={2025},
      booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM’25)}
  }
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see individual module requirements)
- API keys for OpenAI/Qwen models (for classification tasks)

### Installation

```bash
# Clone the repository
git clone https://github.com/mgao97/FediData.git
cd FediData
pip install requirements.txt
```


for example: in the data collection module,
pip install -r data_collection/userprofile_ugc_download/requirements.txt


## 📁 Repository Structure

```
FediData/
├── 📂 data_collection/           # Data collection tools
│   ├── 📂 image_download/       # Image downloading utilities
│   └── 📂 userprofile_ugc_download/  # User and post data collection
├── 📂 bot_detection/            # Social bot detection models
├── 📂 image_category_classification/  # Image classification tools
├── 📂 topic_emotion/            # Topic and emotion analysis
├── 📂 dataset/                  # Raw and processed data
└── README.md                   # Project overview and usage guide
```

## 🛠️ Usage Guide

This repository contains multiple modules for different aspects of Mastodon data processing. Each module has its own detailed README with specific usage instructions:

### Data Collection
- **[User Profile & UGC Collection](data_collection/userprofile_ugc_download/readme.md)**: Complete pipeline for collecting user profiles, social networks, and posts from Mastodon instances
- **[Image Download](data_collection/image_download/README.md)**: Concurrent image downloader for extracting images from collected posts
- You might either collect the data using our provided code or directly download the anonymized dataset from Zenodo and extract it into the dataset folder.
### Data Analysis 
- **[Topic & Emotion Analysis](topic_emotion/readme.md)**: Topic classification and sentiment analysis of posts
- **[Bot Detection](bot_detection/readme.md)**: Social bot detection using multiple machine learning models
- **[Image Classification](image_category_classification/README.md)**: Automated image categorization using vision-language models

> 💡 **Tip**: Each module's README contains detailed prerequisites, configuration steps, and usage examples.


## Analysis Details

### Topic & Emotion Analysis

- **Topic Classification**: Automated topic categorization using LLMs
- **Emotion Analysis**: Sentiment and emotion detection in posts
- **Visualization**: Comprehensive charts and comparative analysis

### Bot Detection Models

| Model | Description | Type |
|-------|-------------|------|
| **BECE** | Bot detection using embedding and classification | Deep Learning |
| **BotRGCN** | Relational Graph Convolutional Network | Graph Neural Network |
| **SGBot** | Statistical and graph-based features | Random Forest |

### Image Classification

- **Qwen 2.5 VL-32B Instruct**: Vision-language model for image categorization
- Supports batch processing with configurable thread pools
- Automatic retry and error handling
