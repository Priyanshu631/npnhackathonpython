import os
import glob
import logging
import joblib
import xgboost as xgb

logger = logging.getLogger("fraud_triage")

class ModelArtifacts:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.models = {}
        self.load_artifacts()

    def _safe_load(self, filename: str):
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return None
        
        # 1. Try standard joblib
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict):
                return obj.get("model") or obj.get("pipeline") or obj
            return obj
        except Exception as e1:
            # 2. Try native XGBoost loader if it's an XGB file
            if "xgb" in filename.lower():
                try:
                    clf = xgb.XGBClassifier()
                    clf.load_model(path)
                    return clf
                except Exception as e2:
                    logger.error(f"Error loading {filename}: joblib({e1}) | xgb_native({e2})")
                    return None
            logger.error(f"Error loading {filename}: {e1}")
            return None

    def load_artifacts(self):
        self.models["baseline_lr"] = self._safe_load("structured_baseline_lr_model.pkl")
        self.models["baseline_tfidf"] = self._safe_load("text_baseline_tf_idf_lr_model.pkl")
        self.models["rf_structured"] = self._safe_load("rf_structured_model.pkl")
        self.models["xgb_structured"] = self._safe_load("xgb_structured_model.pkl")
        self.models["rf_fusion"] = self._safe_load("rf_early_fusion_finbert.pkl")
        self.models["xgb_fusion"] = self._safe_load("xgb_early_fusion_finbert.pkl")

artifacts = ModelArtifacts()