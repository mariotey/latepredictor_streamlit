"""
streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

from utils.supabase_utils import extract_all_rows, SUPABASE_CLIENT
from utils.feature_transform import get_features
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    FEATURES_NAME,
    FEEDBACK_NAME,
    CATEGORY_NAME,
    FEATURE_REGISTRY_NAME,
    MODEL_REGISTRY_NAME,
    FEATURE_REGISTRY_ID,
    MODEL_REGISTRY_ID
)

SG_TZ = "Asia/Singapore"

st.set_page_config(layout="wide")

# Title
st.title("⏱️ Lateness Prediction Monitoring Dashboard")

# Refreshes every hour
st_autorefresh(interval=3600000/4)

def compute_metrics(df: pd.DataFrame):
    if df.empty:
        return {
            "mae": 0,
            "mse": 0,
            "rmse": 0,
            "bias": 0,
            "count": 0
        }

    y_true = df["act_min"]
    y_pred = df["pred_min"]

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "bias": np.mean(y_pred - y_true),
        "count": len(df)
    }

#################################################################################################
# Data Loading

feature_df = extract_all_rows(FEATURES_NAME)
feedback_df = extract_all_rows(FEEDBACK_NAME)
category_df = extract_all_rows(CATEGORY_NAME)

feature_registry_dict = (
    SUPABASE_CLIENT.table(FEATURE_REGISTRY_NAME)
    .select("config")
    .eq("f_reg_id", FEATURE_REGISTRY_ID)
    .single()
    .execute()
).data["config"]

model_registry_dict = (
    SUPABASE_CLIENT.table(MODEL_REGISTRY_NAME)
    .select("*")
    .eq("model_id", MODEL_REGISTRY_ID)
    .execute()
).data[0]

numerical_cols = feature_registry_dict["feature_col"]["numerical"]
categorical_cols = feature_registry_dict["feature_col"]["categorical"]

#################################################################################################
# Preprocessing

# Process the user input into model features
feedback_features_df = get_features(feedback_df)
feedback_features_df = (
    feedback_features_df
    .merge(category_df, on="category_id", how="left")
    .drop(columns=["category_id"])
)

error_df = feedback_features_df.merge(
    feedback_df[["feedback_id", "date"]],
    on="feedback_id",
    how="left"
)

error_df["error"] = error_df["act_min"] - error_df["pred_min"]
error_df["abs_error"] = error_df["error"].abs()

def compute_metrics(df):
    if df.empty:
        return {"mae": 0, "mse": 0, "rmse": 0, "bias": 0, "p90": 0, "count": 0}

    y_true = df["act_min"]
    y_pred = df["pred_min"]

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "bias": np.mean(y_pred - y_true),
        "count": len(df)
    }

metrics = compute_metrics(feedback_features_df)
train_rmse = np.sqrt(model_registry_dict["mse"])

#################################################################################################
# Overview Section
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registry ID of Features Used", FEATURE_REGISTRY_ID)
col2.metric("Registry ID of Model Used", MODEL_REGISTRY_ID)
col3.metric("No. of Training Data", len(feature_df))
col4.metric("No. of Feedback Data", metrics["count"])

#################################################################################################

st.subheader("📦 Data Drift")

cols = st.columns(len(numerical_cols))

for i, col in enumerate(numerical_cols):
    if col not in feature_df.columns or col not in feedback_features_df.columns:
        continue

    df_plot = pd.concat([
        feature_df.assign(dataset="train"),
        feedback_features_df.assign(dataset="feedback")
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

cols = st.columns(len(categorical_cols))

for i, col in enumerate(categorical_cols):
    if col not in feature_df.columns or col not in feedback_features_df.columns:
        continue

    train_dist = feature_df[col].value_counts(normalize=True)
    feedback_dist = feedback_features_df[col].value_counts(normalize=True)

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

#################################################################################################

st.subheader("📉 Model Drift")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{metrics['mae']:.2f}")
col2.metric("MSE", f"{metrics['mse']:.2f}")
col3.metric("RMSE", f"{metrics['rmse']:.2f}")
col4.metric("Bias", f"{metrics['bias']:.2f}")

fig1 = px.scatter(
    feedback_features_df,
    x="pred_min",
    y="act_min",
    labels={
        "pred_min": "Predicted Lateness (minutes)",
        "act_min": "Actual Lateness (minutes)"
    }
)

# Perfect prediction reference line
min_val = min(feedback_features_df["pred_min"].min(), feedback_features_df["act_min"].min())
max_val = max(feedback_features_df["pred_min"].max(), feedback_features_df["act_min"].max())

fig1.add_shape(
    type="line",
    x0=min_val,
    x1=max_val,
    y0=min_val,
    y1=max_val,
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig1, width="stretch")

train_rmse = np.sqrt(model_registry_dict["mse"])

time_df = error_df.groupby("date").agg(
    rmse=("error", lambda x: (x**2).mean() ** 0.5)
).reset_index()

fig = px.line(
    time_df,
    x="date",
    y="rmse",
    markers=True,
    title="RMSE Over Time (Train vs Live)"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Root Mean Squared Error (minutes)",
)

fig.add_hline(
    y=train_rmse,
    line_dash="dot",
    line_color="red",
    annotation_text="Train RMSE",
    annotation_position="top right"
)

st.plotly_chart(fig, width="stretch")

#################################################################################################

st.subheader("🚨 Worst Predictions")

st.dataframe(
    (
        error_df
        .sort_values("abs_error", ascending=False)
        .head(10)[["feedback_id"] + categorical_cols + numerical_cols + ["act_min", "pred_min"]]
        .reset_index(drop=True)
    ),
    width="stretch"
)