# Dataset 3 Fraud Intelligence Dashboard

A Streamlit dashboard for the final Dataset 3 fraud-detection model suite.

## Model families

### Structured only
- Logistic Regression
- Random Forest
- XGBoost

### Text only
- TF-IDF + Logistic Regression
- TF-IDF + Random Forest
- TF-IDF + XGBoost
- FinBERT + Logistic Regression

### Both
- RF + TF-IDF late fusion
- XGBoost + TF-IDF late fusion
- RF + FinBERT late fusion
- XGBoost + FinBERT late fusion
- XGBoost + FinBERT + Isolation Forest (Triumvirate)

Early-fusion models and the old fusion baseline are intentionally not included.

## Folder layout

Put the exported main model suite into `model_outputs/`:

```text
model_outputs/
  structured_baseline.joblib
  text_baseline_tfidf_lr.joblib
  rf_structured.joblib
  rf_text.joblib
  xgb_structured.joblib
  xgb_text.joblib
  rf_late_fusion.joblib
  xgb_late_fusion.joblib
  finbert_text.joblib
  rf_late_fusion_finbert.joblib
  xgb_late_fusion_finbert.joblib
  baseline_metrics_updated.json
  late_fusion_metrics.json
  transformer_fusion_metrics.json
  final_ranked_metrics.json
  model_manifest.json
  deployment_metadata.json
  requirements.txt
```

Put the edited Triumvirate notebook output into `triumvirate_outputs/`:

```text
triumvirate_outputs/
  triumvirate_fusion.joblib
  triumvirate_fusion_metrics.json
  triumvirate_fusion_metadata.json
  triumvirate_model_manifest.json
```

Put `dataset3_gold_with_notes_wild.csv` beside `app.py`.

The app also supports environment variables if your artifacts live elsewhere:

```text
DATASET3_MODEL_DIR=/path/to/model_outputs
DATASET3_TRIUMVIRATE_DIR=/path/to/triumvirate_outputs
DATASET3_DATA_PATH=/path/to/dataset3_gold_with_notes_wild.csv
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first prediction involving FinBERT downloads `ProsusAI/finbert` and then caches it for the Streamlit process.
