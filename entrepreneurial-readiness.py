import json
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from xgboost import XGBClassifier

REPO_ID = "mjpsm/Entrepreneurial-Readiness-XGB-v2"

@st.cache_resource
def load_model_and_meta():
    model_file = hf_hub_download(REPO_ID, "xgb_model.json")
    feat_file  = hf_hub_download(REPO_ID, "feature_order.json")
    map_file   = hf_hub_download(REPO_ID, "label_map.json")

    clf = XGBClassifier()
    clf.load_model(model_file)
    feature_order = json.load(open(feat_file))
    label_map = json.load(open(map_file))        # e.g., {"low":0,"medium":1,"high":2}
    inv_map = {v: k for k, v in label_map.items()}
    return clf, feature_order, label_map, inv_map

def add_derived(r: dict) -> dict:
    """Compute savings_to_expense_ratio (cap 12) and runway_months (cap 60) if missing."""
    bills   = float(r["monthly_bills"])
    ent     = float(r["monthly_entertainment_spend"])
    income  = float(r["monthly_income"])
    savings = float(r["savings"])

    if r.get("savings_to_expense_ratio") in (None, ""):
        denom = bills + ent
        r["savings_to_expense_ratio"] = min(12.0, (savings / denom) if denom > 0 else 12.0)

    if r.get("runway_months") in (None, ""):
        net_burn = (bills + ent) - income
        r["runway_months"] = 60.0 if net_burn <= 0 else max(0.0, min(60.0, savings / net_burn))
    return r

st.set_page_config(page_title="Entrepreneurial Readiness", page_icon="💼", layout="centered")

st.title("Entrepreneurial Readiness • XGBoost")
st.caption("Model: mjpsm/Entrepreneurial-Readiness-XGB-v2")

clf, feature_order, label_map, inv_map = load_model_and_meta()

with st.form("inputs"):
    st.subheader("Financial")
    col1, col2 = st.columns(2)
    with col1:
        savings = st.number_input("Savings ($)", min_value=0.0, value=0.0, step=100.0)
        monthly_bills = st.number_input("Monthly Bills ($)", min_value=0.0, value=0.0, step=50.0)
        assets = st.number_input("Assets ($)", min_value=0.0, value=0.0, step=100.0)
    with col2:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=0.0, step=50.0)
        monthly_entertainment_spend = st.number_input("Monthly Discretionary Spend ($)", min_value=0.0, value=0.0, step=25.0)

    st.subheader("Personal & Skills")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=0, value=25, step=1)
        sales = st.slider("Sales Skills (1–10)", 1, 10, 5)
    with c2:
        dependents = st.number_input("Dependents", min_value=0, value=0, step=1)
        risk = st.slider("Risk Tolerance (1–10)", 1, 10, 5)
    with c3:
        hours = st.number_input("Available Hours/Week", min_value=0, value=20, step=1)
        confidence = st.slider("Confidence (1–10)", 1, 10, 5)

    idea_difficulty = st.slider("Idea Difficulty (1–10; higher = harder)", 1, 10, 5)

    st.subheader("Experience")
    d1, d2 = st.columns(2)
    with d1:
        prior_started = st.number_input("Prior Businesses Started", min_value=0, value=0, step=1)
    with d2:
        prior_exits = st.number_input("Prior Exits", min_value=0, value=0, step=1)

    with st.expander("Advanced (optional) — leave blank to auto-compute"):
        runway_months = st.text_input("Runway (months)", value="")
        ser = st.text_input("Savings-to-Expense Ratio", value="")

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "savings": savings,
        "monthly_income": monthly_income,
        "monthly_bills": monthly_bills,
        "monthly_entertainment_spend": monthly_entertainment_spend,
        "sales_skills_1to10": float(sales),
        "age": float(age),
        "dependents_count": float(dependents),
        "assets": assets,
        "risk_tolerance_1to10": float(risk),
        "confidence_1to10": float(confidence),
        "idea_difficulty_1to10": float(idea_difficulty),
        "prior_businesses_started_": float(prior_started),
        "prior_exits": float(prior_exits),
        "time_available_hours_per_week": float(hours),
        "savings_to_expense_ratio": float(ser) if ser.strip() else None,
        "runway_months": float(runway_months) if runway_months.strip() else None,
    }

    row = add_derived(row)

    # Build DataFrame in exact order expected by the model
    try:
        X = pd.DataFrame([row])[feature_order]
    except KeyError as e:
        st.error(f"Missing feature for model: {e}")
        st.stop()

    pred_id = int(clf.predict(X)[0])
    probs = clf.predict_proba(X)[0].tolist()
    pred_label = inv_map[pred_id]

    st.success(f"**Prediction:** {pred_label.upper()}")
    st.write("**Class probabilities:**")
    for i, p in enumerate(probs):
        label = inv_map[i]
        st.progress(min(max(p, 0.0), 1.0), text=f"{label} — {p:.3f}")

    with st.expander("Inputs used (after auto-compute)"):
        st.json(row)
