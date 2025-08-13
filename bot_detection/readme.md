# Social Bot Detection
Social bot detection based on our FediData dataset leverages user profiles, posts, images, and social connections to accurately identify social bots. We have implemented several representative baselines.

## Files
- data_split.py : Script for splitting the original dataset into training, validation, and test sets.
- data_vis.py : Script for visualizing data distributions and feature analysis.
- feature_extract.py : Script for extracting numerical, categorical, text, and image features from raw data.
- final_results.py : Script for aggregating the final results from different models.
- top_bottom3_results.py : Script for summarizing the top and bottom 3 performing instances for each model.

## Folders
- figs/ : Stores various visualization charts and analysis result images.
- models/ : Contains model-related code.
  - BECE.py : Implementation of the BECE model.
  - BECE_train.py : Training and evaluation workflow for the BECE model.
  - BotRGCN.py : Implementation of the BotRGCN model.
  - BotRGCN_train.py : Training and evaluation workflow for the BotRGCN model.
  - sgbot_model.py : Training and evaluation for the SGBot model (Random Forest based).
  - xgboost_model.py : Training and evaluation for the XGBoost model.
- results/ : Stores test results (CSV files) for each model.
  - bece_test_results.csv : BECE model test results.
  - botrgcn_test_results.csv : BotRGCN model test results.
  - sgbot_test_results.csv : SGBot model test results.
  - xgboost_test_results.csv : XGBoost model test results.

## Usage
### Data Preparation
1. Use `data_split.py` to split the raw dataset into training, validation, and test sets.
2. Extract features using `feature_extract.py`.
3. Visualize data distributions and feature statistics with `data_vis.py`.

### Model Training and Evaluation

- Train and evaluate the BECE model using `BECE_train.py`.
- Train and evaluate the BotRGCN model using `BotRGCN_train.py`.
- Train and evaluate the SGBot model using `sgbot_model.py`.
- Train and evaluate the XGBoost model using `xgboost_model.py`.

### Results Analysis

- Aggregate final model results with `final_results.py`.
- Summarize top and bottom 3 performing instances per model using `top_bottom3_results.py`.

---
