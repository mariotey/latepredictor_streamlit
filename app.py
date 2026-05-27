"""
streamlit run app.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from utils.supabase_utils import extract_all_rows
from utils.feature_transform import get_features
from utils.statistics_utils import compute_metrics
from config import (
    FEATURES_NAME,
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
feature_df = extract_all_rows(FEATURES_NAME)
feedback_df = extract_all_rows(FEEDBACK_NAME)
category_df = extract_all_rows(CATEGORY_NAME)

numeric_cols = ["distance_km", "day_of_week"]
categorical_cols = ["time_of_day", "category"]

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

col1, col2 = st.columns(2)

col1.metric("No. of Training", len(feature_df))
col2.metric("No. of Feedback", metrics["count"])

st.subheader("Numeric Features Drift")

cols = st.columns(len(numeric_cols))

for i, col in enumerate(numeric_cols):
    if col not in feature_df.columns or col not in modified_feedback_df.columns:
        continue

    df_plot = pd.concat([
        feature_df.assign(dataset="train"),
        modified_feedback_df.assign(dataset="feedback")
    ])

    nbins = 10 if col == "day_of_week" else 50

    fig = px.histogram(
        df_plot,
        x=col,
        color="dataset",
        barmode="overlay",
        opacity=0.5,
        nbins=nbins,
        histnorm="probability density",
        title=col
    )

    with cols[i]:
        st.plotly_chart(fig, width="stretch")

st.subheader("Categorical Features Drift")

cols = st.columns(len(categorical_cols))

for i, col in enumerate(categorical_cols):
    if col not in feature_df.columns or col not in modified_feedback_df.columns:
        continue

    train_dist = feature_df[col].value_counts(normalize=True)
    feedback_dist = modified_feedback_df[col].value_counts(normalize=True)

    all_cats = sorted(set(train_dist.index).union(feedback_dist.index))

    df_plot = pd.DataFrame({
        "category": all_cats,
        "train": [train_dist.get(x, 0) for x in all_cats],
        "feedback": [feedback_dist.get(x, 0) for x in all_cats]
    })

    df_plot = df_plot.melt(
        id_vars="category",
        var_name="dataset",
        value_name="proportion"
    )

    fig = px.bar(
        df_plot,
        x="category",
        y="proportion",
        color="dataset",
        barmode="overlay",
        opacity=0.6,
        title=f"{col}"
    )

    with cols[i]:
        st.plotly_chart(fig, width="stretch")

st.subheader("Actual VS Predicted")

fig1 = px.scatter(
    modified_feedback_df,
    x="pred_min",
    y="act_min",
    labels={
        "pred_min": "Predicted Lateness (minutes)",
        "act_min": "Actual Lateness (minutes)"
    }
)

# Perfect prediction reference line
min_val = min(modified_feedback_df["pred_min"].min(), modified_feedback_df["act_min"].min())
max_val = max(modified_feedback_df["pred_min"].max(), modified_feedback_df["act_min"].max())

fig1.add_shape(
    type="line",
    x0=min_val,
    x1=max_val,
    y0=min_val,
    y1=max_val,
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig1, width="stretch")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", f"{metrics['mae']:.2f}")
col2.metric("MSE", f"{metrics['mse']:.2f}")
col3.metric("RMSE", f"{metrics['rmse']:.2f}")
col4.metric("Bias", f"{metrics['bias']:.2f}")
col5.metric("P90 Error", f"{metrics['p90_error']:.2f}")

# # CHART 2: ERROR DISTRIBUTION
# fig2 = px.histogram(
#     feedback_df,
#     x="error",
#     nbins=30,
#     labels={"error": "Prediction Error (minutes)"},
#     title="Distribution of Prediction Errors"
# )

# st.plotly_chart(fig2, width="stretch")


# Top 10 Worst Predictions
error_feedback_df = modified_feedback_df.copy()
error_feedback_df["error"] = error_feedback_df["act_min"] - error_feedback_df["pred_min"]
error_feedback_df["abs_error"] = abs(error_feedback_df["error"])

st.subheader("Top 10 Worst Predictions")

st.dataframe(
    (
        error_feedback_df
        .sort_values("abs_error", ascending=False)
        .head(10)[["feedback_id"] + categorical_cols + numeric_cols + ["act_min", "pred_min"]]
        .reset_index()
    ),
    width="stretch"
)

# TODO: Model Drift and Concept Drift Time Series Plot
