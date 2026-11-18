
import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="PARLAY KINGZ",
    page_icon="👑",
    layout="wide"
)

import joblib
from pathlib import Path

# =====================================
# PARLAY KINGZ WEBSITE
# =====================================

st.set_page_config(
    page_title="PARLAY KINGZ",
    page_icon="👑",
    layout="wide"
)

# =====================================
# PRIVATE MODEL BACKUP
# =====================================

BACKUP_PATH = (
    Path(__file__).parent /
    "PARLAY_KINGZ_PRIVATE_BACKUP.joblib"
)

# =====================================
# WEBSITE DESIGN
# =====================================

st.markdown("""
<style>
.stApp {
    background-color: #10131b;
    color: white;
}
h1 {
    color: #e5b93f;
    text-align: center;
}
h2, h3 {
    color: #e5b93f;
}
</style>
""", unsafe_allow_html=True)

st.title("👑 PARLAY KINGZ")
st.markdown(
    "<p style='text-align:center;'>"
    "COLLEGE BASKETBALL MODEL"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# =====================================
# LOAD EXISTING MODEL
# =====================================

@st.cache_resource
def load_model():
    backup = joblib.load(BACKUP_PATH)

    return (
        backup["base_model"],
        backup["calibrator"],
        backup["raw_scores"],
        backup["safe_games"],
        backup["master_ledger"],
        backup["features"]
    )

if not BACKUP_PATH.exists():
    st.error("Model backup file not found.")
    st.stop()

try:
    model, calibrator, raw_scores, safe_games, ledger, features = load_model()

    if model.n_features_in_ != len(features):
        st.error("Model feature mismatch.")
        st.stop()

    st.success("🟢 PARLAY KINGZ MODEL ONLINE")

except Exception as error:
    st.error("Unable to load the existing model.")
    st.code(str(error))
    st.stop()

# =====================================
# DASHBOARD
# =====================================

col1, col2, col3 = st.columns(3)

col1.metric("MODEL STATUS", "ONLINE")
col2.metric("MODEL FEATURES", len(features))
col3.metric("HISTORICAL GAMES", f"{len(safe_games):,}")

st.divider()

# =====================================
# MODEL INFORMATION
# =====================================

st.header("📊 Prediction Engine")

st.write("Model: Random Forest Classifier")
st.write("Calibration: Sigmoid / Logistic Regression")
st.write("Training: Existing trained model")
st.write("Automatic retraining: Disabled")

with st.expander("View Model Features"):
    for feature in features:
        st.write("•", feature)

# =====================================
# HISTORICAL DATA
# =====================================

st.header("📚 Historical Model Data")

st.write("Historical score records:", f"{len(raw_scores):,}")
st.write("Master ledger records:", f"{len(ledger):,}")

# =====================================
# DAILY PICKS - COMING NEXT
# =====================================

st.divider()

st.header("🏀 Daily College Basketball Picks")

st.info(
    "Your existing model is connected. "
    "The live schedule and daily prediction engine "
    "will be added in the next development step."
)

st.caption(
    "PARLAY KINGZ | Private Development Dashboard"
)
