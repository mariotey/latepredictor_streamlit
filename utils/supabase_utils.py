import pandas as pd
import os
import streamlit as st
from supabase import create_client

def get_info():
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    # Check if the values obtained are valid
    if not all([SUPABASE_URL, SUPABASE_KEY]):
        raise ValueError("Missing Supabase configuration")

    return SUPABASE_URL, SUPABASE_KEY

SUPABASE_URL, SUPABASE_KEY = get_info()

# Create client once
SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_all_rows(table_name):
    res = SUPABASE_CLIENT.table(table_name).select("*").execute()

    return pd.DataFrame(res.data)