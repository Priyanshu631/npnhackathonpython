from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "policy_annual_premium",
    "total_claim_amount",
    "age",
    "months_as_customer",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
]

CATEGORICAL_FEATURES = [
    "policy_state",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "police_report_available",
]

TEXT_FEATURE = "adjuster_notes"

MODEL_SPECS = {
    "structured_baseline": {
        "name": "Logistic Regression",
        "group": "Structured",
        "artifact": "structured_baseline.joblib",
        "kind": "structured",
    },
    "rf_structured": {
        "name": "Random Forest",
        "group": "Structured",
        "artifact": "rf_structured.joblib",
        "kind": "structured",
    },
    "xgb_structured": {
        "name": "XGBoost",
        "group": "Structured",
        "artifact": "xgb_structured.joblib",
        "kind": "structured",
    },
    "text_baseline": {
        "name": "TF-IDF + Logistic Regression",
        "group": "Text",
        "artifact": "text_baseline_tfidf_lr.joblib",
        "kind": "text_bundle",
    },
    "rf_text": {
        "name": "TF-IDF + Random Forest",
        "group": "Text",
        "artifact": "rf_text.joblib",
        "kind": "text",
    },
    "xgb_text": {
        "name": "TF-IDF + XGBoost",
        "group": "Text",
        "artifact": "xgb_text.joblib",
        "kind": "text",
    },
    "finbert_text": {
        "name": "FinBERT + Logistic Regression",
        "group": "Text",
        "artifact": "finbert_text.joblib",
        "kind": "finbert_bundle",
    },
    "rf_fusion_lr": {
        "name": "RF + TF-IDF → Logistic Regression",
        "group": "Fusion",
        "artifact": "rf_late_fusion.joblib",
        "kind": "rf_late",
    },
    "xgb_fusion_lr": {
        "name": "XGBoost + TF-IDF → Logistic Regression",
        "group": "Fusion",
        "artifact": "xgb_late_fusion.joblib",
        "kind": "xgb_late",
    },
    "rf_late_fusion_finbert": {
        "name": "RF + FinBERT → Logistic Regression",
        "group": "Fusion",
        "artifact": "rf_late_fusion_finbert.joblib",
        "kind": "rf_finbert_late",
    },
    "xgb_late_fusion_finbert": {
        "name": "XGBoost + FinBERT → Logistic Regression",
        "group": "Fusion",
        "artifact": "xgb_late_fusion_finbert.joblib",
        "kind": "xgb_finbert_late",
    },
    "triumvirate_fusion": {
        "name": "XGBoost + FinBERT + Isolation Forest",
        "group": "Fusion",
        "artifact": "triumvirate_fusion.joblib",
        "kind": "triumvirate",
    },
}

GROUPS = {
    "Structured only": ["structured_baseline", "rf_structured", "xgb_structured"],
    "Text only": ["text_baseline", "rf_text", "xgb_text", "finbert_text"],
    "Both": [
        "rf_fusion_lr",
        "xgb_fusion_lr",
        "rf_late_fusion_finbert",
        "xgb_late_fusion_finbert",
        "triumvirate_fusion",
    ],
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(model_dir: Path, triumvirate_dir: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for filename in (
        "baseline_metrics_updated.json",
        "late_fusion_metrics.json",
        "transformer_fusion_metrics.json",
        "final_ranked_metrics.json",
    ):
        data = _load_json(model_dir / filename)
        if data:
            merged.update(data)
    tri = _load_json(triumvirate_dir / "triumvirate_fusion_metrics.json")
    if tri:
        # The training notebook stores the result under
        # "triumvirate_late_fusion", while the dashboard model ID is
        # "triumvirate_fusion".
        if "triumvirate_late_fusion" in tri:
            merged["triumvirate_fusion"] = tri["triumvirate_late_fusion"]
        elif "triumvirate_fusion" in tri:
            merged["triumvirate_fusion"] = tri["triumvirate_fusion"]
    return merged


def load_manifest(model_dir: Path) -> dict[str, Any]:
    return _load_json(model_dir / "model_manifest.json")


def load_metadata(model_dir: Path) -> dict[str, Any]:
    return _load_json(model_dir / "deployment_metadata.json")


def available_models(model_dir: Path, triumvirate_dir: Path) -> list[str]:
    out = []
    for model_id, spec in MODEL_SPECS.items():
        root = triumvirate_dir if model_id == "triumvirate_fusion" else model_dir
        if (root / spec["artifact"]).exists():
            out.append(model_id)
    return out


def load_artifact(model_dir: Path, triumvirate_dir: Path, model_id: str):
    spec = MODEL_SPECS[model_id]
    root = triumvirate_dir if model_id == "triumvirate_fusion" else model_dir
    path = root / spec["artifact"]
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return joblib.load(path)


def structured_frame(values: dict[str, Any]) -> pd.DataFrame:
    row = {k: values.get(k) for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def _finbert_embedding(text: str, tokenizer, model, torch, device, max_length: int = 128) -> np.ndarray:
    encoded = tokenizer(
        [text or ""],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
        return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()


def predict(
    model_id: str,
    values: dict[str, Any],
    text: str,
    *,
    model_dir: Path,
    triumvirate_dir: Path,
    finbert_loader,
) -> dict[str, Any]:
    artifact = load_artifact(model_dir, triumvirate_dir, model_id)
    spec = MODEL_SPECS[model_id]
    Xs = structured_frame(values)
    text = text or ""

    if spec["kind"] == "structured":
        probability = float(artifact.predict_proba(Xs)[:, 1][0])

    elif spec["kind"] == "text_bundle":
        vector = artifact["vectorizer"]
        model = artifact["model"]
        probability = float(model.predict_proba(vector.transform([text]))[:, 1][0])

    elif spec["kind"] == "text":
        probability = float(artifact.predict_proba(pd.DataFrame({TEXT_FEATURE: [text]}))[:, 1][0])

    elif spec["kind"] == "finbert_bundle":
        tokenizer, finbert, torch, device = finbert_loader()
        emb = _finbert_embedding(text, tokenizer, finbert, torch, device, artifact.get("max_length", 128))
        probability = float(artifact["model"].predict_proba(emb)[:, 1][0])

    elif spec["kind"] in {"rf_late", "xgb_late"}:
        structured_model = artifact["structured_model"]
        text_model = artifact["text_model"]
        meta = artifact["meta_model"]
        p_struct = float(structured_model.predict_proba(Xs)[:, 1][0])
        p_text = float(text_model.predict_proba(pd.DataFrame({TEXT_FEATURE: [text]}))[:, 1][0])
        probability = float(meta.predict_proba([[p_struct, p_text]])[:, 1][0])

    elif spec["kind"] in {"rf_finbert_late", "xgb_finbert_late"}:
        tokenizer, finbert, torch, device = finbert_loader()
        structured_model = artifact["structured_model"]
        text_model = artifact["finbert_text_model"]
        meta = artifact["meta_model"]
        p_struct = float(structured_model.predict_proba(Xs)[:, 1][0])
        emb = _finbert_embedding(text, tokenizer, finbert, torch, device, artifact.get("finbert_max_length", 128))
        p_text = float(text_model.predict_proba(emb)[:, 1][0])
        probability = float(meta.predict_proba([[p_struct, p_text]])[:, 1][0])

    elif spec["kind"] == "triumvirate":
        components = artifact["components"]
        tokenizer, finbert, torch, device = finbert_loader()

        p_xgb = float(components["xgb_structured_base"].predict_proba(Xs)[:, 1][0])
        emb = _finbert_embedding(text, tokenizer, finbert, torch, device, artifact.get("finbert_max_length", 128))
        p_fb = float(components["finbert_text_base"].predict_proba(emb)[:, 1][0])

        iso_values = Xs.copy()
        iso_values["police_report_available"] = iso_values["police_report_available"].apply(
            lambda x: 1 if str(x).upper() == "YES" else 0
        )
        shared = artifact["shared_features"]
        p_anomaly = float(
            -components["isolation_forest"].decision_function(
                components["if_preprocessor"].transform(iso_values[shared])
            )[0]
        )
        probability = float(components["meta_model"].predict_proba([[p_xgb, p_fb, p_anomaly]])[:, 1][0])
        return {
            "probability": probability,
            "prediction": int(probability >= 0.5),
            "signals": {
                "XGBoost structured": p_xgb,
                "FinBERT text": p_fb,
                "Isolation Forest anomaly": p_anomaly,
            },
        }

    else:
        raise ValueError(f"Unknown inference kind: {spec['kind']}")

    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "signals": {},
    }
