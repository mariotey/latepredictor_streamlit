import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from utils.supabase_utils import extract_all_rows
from utils.feature_transform import get_features
from utils.statistics_utils import compute_metrics
from config import (
    FEATURE_NAME,
    FEEDBACK_NAME,
    CATEGORY_NAME,
    FEATURE_REGISTRY_NAME,
    FEATURE_REGISTRY_ID
)

st.set_page_config(layout="wide")

# Title
st.title("⏱️ Lateness Prediction Monitoring Dashboard")

# Refreshes every hour
st_autorefresh(interval=3600000/4)

# Import Data
feature_df = extract_all_rows(FEATURE_NAME)
feedback_df = extract_all_rows(FEEDBACK_NAME)
category_df = extract_all_rows(CATEGORY_NAME)

# Process the user input into model features
modified_feedback_df = get_features(feedback_df)
modified_feedback_df = (
    modified_feedback_df
    .merge(
        category_df,
        how="left",
        on="category_id"
    )
    .drop(columns=["category_id"])
)

# General Metrics
metrics = compute_metrics(modified_feedback_df)

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("MAE", f"{metrics['mae']:.2f}")
col2.metric("MAE", f"{metrics['mse']:.2f}")
col3.metric("RMSE", f"{metrics['rmse']:.2f}")
col4.metric("Bias", f"{metrics['bias']:.2f}")
col5.metric("P90 Error", f"{metrics['p90_error']:.2f}")
col6.metric("Samples", metrics["count"])


# CHART 1: Predicted vs Actual
fig1 = px.scatter(
    feedback_df,
    x="pred_min",
    y="act_min",
    labels={
        "pred_min": "Predicted Lateness (minutes)",
        "act_min": "Actual Lateness (minutes)"
    },
    title="Predicted vs Actual Lateness"
)


# Perfect prediction reference line
min_val = min(feedback_df["pred_min"].min(), feedback_df["act_min"].min())
max_val = max(feedback_df["pred_min"].max(), feedback_df["act_min"].max())

fig1.add_shape(
    type="line",
    x0=min_val,
    x1=max_val,
    y0=min_val,
    y1=max_val,
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig1, width="stretch")


# CHART 2: ERROR DISTRIBUTION
fig2 = px.histogram(
    feedback_df,
    x="error",
    nbins=30,
    labels={"error": "Prediction Error (minutes)"},
    title="Distribution of Prediction Errors"
)

st.plotly_chart(fig2, width="stretch")


# Debug Table
st.subheader("Top 10 Worst Predictions")
feedback_df["abs_error"] = abs(feedback_df["error"])
st.dataframe(feedback_df.sort_values("abs_error", ascending=False).head(10), use_container_width=True)

# TODO: Incorportae MSE
# TODO: Incorporate Data Drift, Model Drift and Concept Drift Distribution