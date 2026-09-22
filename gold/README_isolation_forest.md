# Isolation Forest - Tabular Anomaly Detection

## Purpose
Detect unusual insurance claims from structured/tabular data and produce an anomaly score for investigator triage.

## Dataset
- Records: 30,000
- Fraud rate: 11.47%
- Training records: 24,000
- Test records: 6,000
- Isolation Forest trained on legitimate training claims only: 21,248

## Key EDA findings
- `authorities_contacted = None` had the highest fraud rate: 21.13%.
- `incident_severity = Total Loss` had a higher fraud rate: 14.81%.
- `claim_to_total_ratio` was higher for fraud claims.
- No major univariate numeric outliers were present.
- Claim amount and total claim amount were strongly correlated (0.897).

## Model configuration
- Isolation Forest
- n_estimators: 300
- contamination: 0.1147
- Random state: 42
- Final encoded features: 90

## Evaluation
- ROC-AUC: 0.5065
- Precision: 0.1168
- Recall: 0.1221
- F1-score: 0.1194
- Claims flagged for review: 719

## Interpretation
Isolation Forest identified unusual tabular patterns but did not effectively separate known fraud claims from legitimate claims. Its anomaly score should be used as a supplementary risk signal alongside the later supervised XGBoost/LightGBM fusion model.

## Deliverables
- `claims_tabular_features.parquet`
- `isolation_forest_predictions.parquet`
- `isolation_forest_metrics.parquet`
- `isolation_forest_model.joblib`
- `isolation_forest_preprocessor.joblib`
- `isolation_forest_input_columns.json`
