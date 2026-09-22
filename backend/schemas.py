from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- Claim Prediction Payloads ---
class ClaimInput(BaseModel):
    claim_id: str = Field(..., example="CLM-98214")
    model_name: str = Field(default="FinBERT + XGBoost Fusion")
    claim_amount: float = Field(..., example=5500.0)
    policy_tenure_months: int = Field(..., example=14)
    incident_type: str = Field(..., example="Multi-vehicle Collision")
    police_report_filed: bool = Field(..., example=False)
    claim_narrative: str = Field(..., example="Front bumper and axle damage in parking lot.")

class PredictionOutput(BaseModel):
    claim_id: str
    model_used: str
    fraud_probability: float
    risk_tier: str
    anomaly_score: float
    is_anomaly: bool
    feature_importance: Dict[str, float]

# --- Visual Metric Payloads ---
class ModelMetric(BaseModel):
    model_name: str
    accuracy: float
    f1_score: float
    fraud_recall: float
    false_positive_rate: float
    latency_ms: float

class ConfusionMatrixResponse(BaseModel):
    model_name: str
    matrix: List[List[int]]  # [[TN, FP], [FN, TP]]

# --- DuckDB Triage Queue Schemas ---
class TriageQueueItem(BaseModel):
    claim_id: str
    claim_amount: float
    supervised_risk: str
    risk_score: float
    anomaly_score: float
    status: str