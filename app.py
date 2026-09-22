from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inference import (
    CATEGORICAL_FEATURES,
    GROUPS,
    MODEL_SPECS,
    NUMERIC_FEATURES,
    TEXT_FEATURE,
    available_models,
    load_manifest,
    load_metadata,
    load_metrics,
    predict,
)

st.set_page_config(
    page_title="BlackBox • Fraud Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("DATASET3_MODEL_DIR", BASE_DIR / "model_outputs"))
TRIUMVIRATE_DIR = Path(os.getenv("DATASET3_TRIUMVIRATE_DIR", BASE_DIR / "triumvirate_outputs"))
DATA_PATH = Path(os.getenv("DATASET3_DATA_PATH", BASE_DIR / "dataset3_gold_with_notes_wild.csv"))

# ---------- Styling ----------
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .hero { padding: 1.2rem 0 0.6rem 0; }
    .hero h1 { font-size: 2.35rem; letter-spacing: -0.04em; margin-bottom: .25rem; }
    .hero p { color: #64748b; font-size: 1rem; margin-top: 0; }
    .metric-card { border: 1px solid rgba(100,116,139,.18); border-radius: 14px; padding: 18px; background: rgba(248,250,252,.65); }
    .risk-high { color: #b91c1c; font-weight: 700; }
    .risk-low { color: #047857; font-weight: 700; }
    .muted { color: #64748b; }
    div[data-testid="stMetric"] { border: 1px solid rgba(100,116,139,.16); padding: 12px; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data

def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data

def get_metrics():
    return load_metrics(MODEL_DIR, TRIUMVIRATE_DIR)


@st.cache_resource

def finbert_loader():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, torch, device


def fmt_pct(x):
    return f"{float(x) * 100:.1f}%"


def model_table(metrics, model_ids):
    rows = []
    for model_id in model_ids:
        if model_id not in metrics:
            continue
        m = metrics[model_id]
        rows.append(
            {
                "Model": MODEL_SPECS[model_id]["name"],
                "Accuracy": m.get("accuracy"),
                "F1": m.get("f1_score"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "ROC-AUC": m.get("roc_auc"),
                "PR-AUC": m.get("pr_auc"),
                "Latency (s)": m.get("inference_latency_sec"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]:
            df[col] = df[col].astype(float)
        df["Latency (s)"] = df["Latency (s)"].astype(float)
    return df


def available_for_group(group_name, available):
    return [m for m in GROUPS[group_name] if m in available]


metrics = get_metrics()
available = available_models(MODEL_DIR, TRIUMVIRATE_DIR)
data = load_data()

# ---------- Sidebar ----------
st.sidebar.title("BlackBox")
st.sidebar.caption("Multimodal fraud detection workspace")

modality = st.sidebar.radio(
    "Prediction modality",
    ["Structured only", "Text only", "Both"],
    index=2,
    help="Controls which model families are available in the prediction workspace.",
)

threshold = st.sidebar.slider(
    "Fraud threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01,
    help="Probability at or above this threshold is classified as Fraud.",
)

st.sidebar.divider()
st.sidebar.write("**Artifacts**")
st.sidebar.write(f"Main models: `{len([m for m in available if m != 'triumvirate_fusion'])}/12`")
st.sidebar.write(f"Triumvirate: `{'ready' if 'triumvirate_fusion' in available else 'missing'}`")

if not MODEL_DIR.exists():
    st.sidebar.error(f"Missing model directory: {MODEL_DIR}")
if not TRIUMVIRATE_DIR.exists():
    st.sidebar.warning(f"Triumvirate directory not found: {TRIUMVIRATE_DIR}")

# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>Fraud Intelligence Dashboard</h1><p>Multiple models • structured signals, adjuster notes, and multimodal fusion</p></div>',
    unsafe_allow_html=True,
)

# ---------- Overview ----------
tab_overview, tab_models, tab_predict = st.tabs(["Overview", "Model Lab", "Prediction Workspace"])

with tab_overview:
    st.subheader("Model landscape")
    all_ids = [m for m in MODEL_SPECS if m in metrics and m in available]
    table = model_table(metrics, all_ids)

    if table.empty:
        st.error("No model metrics/artifacts were found. Put the main model_outputs folder beside app.py and the triumvirate outputs in triumvirate_outputs/.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Models available", len(table))
        c2.metric("Structured", len([m for m in available if MODEL_SPECS[m]["group"] == "Structured"]))
        c3.metric("Text", len([m for m in available if MODEL_SPECS[m]["group"] == "Text"]))
        c4.metric("Fusion", len([m for m in available if MODEL_SPECS[m]["group"] == "Fusion"]))

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                table.sort_values("F1"),
                x="F1",
                y="Model",
                orientation="h",
                title="F1 score across deployed models",
                text_auto=".3f",
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                table,
                x="Recall",
                y="Precision",
                size="ROC-AUC",
                color="Model",
                hover_data=["F1", "PR-AUC", "Latency (s)"],
                title="Precision vs recall",
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Evaluation table")
        display = table.copy()
        for col in ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]:
            display[col] = display[col].map(lambda x: f"{x:.3f}")
        display["Latency (s)"] = display["Latency (s)"].map(lambda x: f"{x:.3f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.caption("Metrics are the exported held-out test-set metrics from the training notebooks. They are not recalculated by the dashboard.")

with tab_models:
    st.subheader("Model Lab")
    group = st.selectbox("Compare model family", ["Structured only", "Text only", "Both"])
    group_ids = available_for_group(group, available)
    group_table = model_table(metrics, group_ids)

    if group_table.empty:
        st.warning("No artifacts from this family are currently available.")
    else:
        metric_choice = st.selectbox("Chart metric", ["F1", "ROC-AUC", "PR-AUC", "Accuracy", "Recall", "Precision"])
        chart = group_table.sort_values(metric_choice)
        fig = px.bar(chart, x=metric_choice, y="Model", orientation="h", text_auto=".3f", title=f"{metric_choice} — {group}")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

        selected_model = st.selectbox(
            "Inspect model",
            group_ids,
            format_func=lambda x: MODEL_SPECS[x]["name"],
        )
        m = metrics.get(selected_model, {})
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", fmt_pct(m.get("accuracy", 0)))
        b.metric("F1", fmt_pct(m.get("f1_score", 0)))
        c.metric("Precision", fmt_pct(m.get("precision", 0)))
        d.metric("Recall", fmt_pct(m.get("recall", 0)))
        e.metric("PR-AUC", fmt_pct(m.get("pr_auc", 0)))

        cm = m.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(cm, index=["Actual Legit", "Actual Fraud"], columns=["Predicted Legit", "Predicted Fraud"])
            fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix")
            st.plotly_chart(fig, use_container_width=True)

        coeffs = m.get("lr_coefficients")
        if coeffs:
            coef_df = pd.DataFrame({"Signal": list(coeffs.keys()), "Coefficient": list(coeffs.values())})
            fig = px.bar(coef_df, x="Coefficient", y="Signal", orientation="h", title="Meta-model signal coefficients")
            st.plotly_chart(fig, use_container_width=True)

        words = m.get("top_risk_words")
        if words:
            word_df = pd.DataFrame({"Term": list(words.keys()), "Weight": list(words.values())}).sort_values("Weight", ascending=True)
            fig = px.bar(word_df, x="Weight", y="Term", orientation="h", title="Exported high-risk text terms")
            st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("Prediction Workspace")
    st.caption("Enter a claim record and run the models allowed by the selected modality.")

    if modality == "Text only":
        st.info("Structured fields are ignored by the selected text-only models.")
    elif modality == "Structured only":
        st.info("Adjuster notes are ignored by the selected structured models.")
    else:
        st.info("Both structured fields and adjuster notes are used. The Triumvirate model combines XGBoost, FinBERT, and an Isolation Forest anomaly signal.")

    if data.empty:
        st.warning("Dataset 3 CSV was not found. Categorical dropdowns will use fallback values.")

    options = {}
    for col in CATEGORICAL_FEATURES:
        if col in data.columns:
            vals = sorted(data[col].dropna().astype(str).unique().tolist())
        else:
            vals = ["Unknown"]
        options[col] = vals

    # Keep the input state available across Streamlit reruns.
    values = {}
    notes = ""

    # ------------------------------------------------------------
    # Structured inputs: visible only for Structured / Both
    # ------------------------------------------------------------
    if modality in ("Structured only", "Both"):
        st.markdown("### Structured claim details")

        ncols = 3
        cols = st.columns(ncols)

        for i, feature in enumerate(NUMERIC_FEATURES):
            with cols[i % ncols]:
                default = float(data[feature].median()) if feature in data.columns else 0.0

                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=default,
                    step=1.0,
                    format="%.2f",
                    key=f"pred_num_{feature}",
                )

        st.markdown("### Categorical details")

        cols = st.columns(3)

        for i, feature in enumerate(CATEGORICAL_FEATURES):
            with cols[i % 3]:
                values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options[feature],
                    key=f"pred_cat_{feature}",
                )

    # ------------------------------------------------------------
    # Text input: visible only for Text / Both
    # ------------------------------------------------------------
    if modality in ("Text only", "Both"):
        st.markdown("### Adjuster notes")

        notes = st.text_area(
            "adjuster_notes",
            height=180,
            placeholder="Enter the adjuster's narrative / claim assessment notes...",
            key="prediction_adjuster_notes",
        )

    # ------------------------------------------------------------
    # Single prediction action
    # ------------------------------------------------------------
    submitted = st.button(
        "Run selected models",
        type="primary",
        use_container_width=True,
        key="run_selected_models",
    )

    if submitted:
        group_ids = available_for_group(modality, available)
        missing = [m for m in GROUPS[modality] if m not in available]

        if missing:
            st.warning(
                "Unavailable artifacts: "
                + ", ".join(MODEL_SPECS[m]["name"] for m in missing)
            )

        if not group_ids:
            st.error(
                "No models are available for this modality. "
                "Check your model_outputs folders."
            )
        elif modality in ("Text only", "Both") and not notes.strip():
            st.error("Please enter adjuster notes for text or fusion prediction.")
        elif modality in ("Structured only", "Both") and not values:
            st.error("Please enter the structured claim information.")
        else:
            results = []
            for model_id in group_ids:
                try:
                    out = predict(
                        model_id,
                        values,
                        notes,
                        model_dir=MODEL_DIR,
                        triumvirate_dir=TRIUMVIRATE_DIR,
                        finbert_loader=finbert_loader,
                    )
                    p = out["probability"]
                    results.append(
                        {
                            "id": model_id,
                            "model": MODEL_SPECS[model_id]["name"],
                            "probability": p,
                            "label": "FRAUD" if p >= threshold else "LEGITIMATE",
                            "signals": out.get("signals", {}),
                        }
                    )
                except Exception as exc:
                    st.error(
                        f"{MODEL_SPECS[model_id]['name']}: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if results:
                st.markdown("### Prediction results")
                result_df = pd.DataFrame(
                    [{"Model": r["model"], "Fraud probability": r["probability"], "Prediction": r["label"]} for r in results]
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Models executed", len(results))
                c2.metric("Average fraud probability", f"{result_df['Fraud probability'].mean() * 100:.1f}%")
                c3.metric("Models flagging fraud", int((result_df["Fraud probability"] >= threshold).sum()))

                fig = px.bar(
                    result_df.sort_values("Fraud probability"),
                    x="Fraud probability",
                    y="Model",
                    orientation="h",
                    color="Prediction",
                    text=result_df.sort_values("Fraud probability")["Fraud probability"].map(lambda x: f"{x:.1%}"),
                    title="Fraud probability by model",
                )
                fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"Threshold {threshold:.0%}")
                fig.update_xaxes(range=[0, 1], tickformat=".0%")
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

                show_df = result_df.copy()
                show_df["Fraud probability"] = show_df["Fraud probability"].map(lambda x: f"{x:.1%}")
                st.dataframe(show_df, use_container_width=True, hide_index=True)

                tri = next((r for r in results if r["id"] == "triumvirate_fusion"), None)
                if tri and tri["signals"]:
                    st.markdown("### Triumvirate signal breakdown")
                    sig = pd.DataFrame({"Signal": list(tri["signals"].keys()), "Value": list(tri["signals"].values())})
                    fig = px.bar(sig, x="Value", y="Signal", orientation="h", text_auto=".3f", title="Inputs to the triumvirate meta-model")
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Classification threshold: {threshold:.0%}. A probability at or above the threshold is displayed as FRAUD.")
