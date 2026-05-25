import streamlit as st
from utils.supabase_utils import extract_all_rows
from config import FEEDBACK_NAME

st.set_page_config(layout="wide")

# Title
st.title("⏱️ Lateness Prediction Monitoring Dashboard")

# Refreshes every hour
st.autorefresh(interval=3600000)

feedback_df = extract_all_rows(FEEDBACK_NAME)

st.dataframe(feedback_df, use_container_width=True, height=600)
