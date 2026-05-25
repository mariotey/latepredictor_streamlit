import streamlit as st
from streamlit_autorefresh import st_autorefresh
from utils.supabase_utils import extract_all_rows
from utils.statistics_utils import compute_metrics
from config import FEEDBACK_NAME

st.set_page_config(layout="wide")

# Title #########################################################################################
left, center, right = st.columns([1, 6, 1])

with center:
    st.title("⏱️ Lateness Prediction Monitoring Dashboard")

#################################################################################################

# Refreshes every hour
st_autorefresh(interval=3600000)

feedback_df = extract_all_rows(FEEDBACK_NAME)

# General Metrics ###############################################################################
metrics = compute_metrics(feedback_df)

left, center, right = st.columns([1, 20, 1])

with center:
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("MAE", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("Bias", f"{metrics['bias']:.2f}")
    col4.metric("P90 Error", f"{metrics['p90_error']:.2f}")
    col5.metric("Samples", metrics["count"])

#################################################################################################
