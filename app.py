# Filename: streamlit_stock_app.py
# Requirements: streamlit, pandas, numpy, joblib, yfinance, plotly, gdown

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
import plotly.express as px
from datetime import datetime
import gdown 

st.set_page_config(page_title="Stock Analysis & Predictor", layout="wide")

# ---------------------- Configuration ----------------------
# Google Drive link for dataset
GDRIVE_LINK = "https://drive.google.com/uc?id=1pzcyhZy5H1NjadZxRQFXzceUCAr4kNgW"
LOCAL_DATA_FILE = "stocks.csv"

# ---------------------- Utilities ----------------------
@st.cache_data
def load_dataset():
    """Download dataset from Google Drive if not present."""
    if not os.path.exists(LOCAL_DATA_FILE):
        st.info("Downloading dataset from Google Drive...")
        gdown.download(GDRIVE_LINK, LOCAL_DATA_FILE, quiet=False)
    df = pd.read_csv(LOCAL_DATA_FILE, parse_dates=True, low_memory=False)
    return df

@st.cache_resource
def load_model_from_paths():
    possible = ["model.pkl"]
    for p in possible:
        if os.path.exists(p):
            try:
                return joblib.load(p), p
            except Exception:
                continue
    return None, None

def prepare_features_for_model(df_row):
    """Simple feature extractor. Replace with your real feature pipeline."""
    x = df_row.copy()
    x = x.select_dtypes(include=[np.number]).fillna(0)
    return x.values.reshape(1, -1)

# ---------------------- App Layout ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Real-time Prediction", "About"])

# Load dataset and model
try:
    df_main = load_dataset()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    df_main = None

model_obj, model_path = load_model_from_paths()

# Allow manual model upload if not found
if not model_obj:
    with st.sidebar.expander("Upload trained model (.pkl)"):
        model_file = st.file_uploader("Upload model.pkl", type=["pkl"], key="upload_model")
        if model_file:
            try:
                model_obj = joblib.load(model_file)
                st.sidebar.success("Model uploaded")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")

# ---------------------- Pages ----------------------
if page == "Home":
    st.title("📈 Welcome to the Stock Safety Predictor")
    st.markdown(
        """
        **Want to buy a stock more safely?**

        Our project analyses historical price behaviour for a curated list of stocks and uses a trained model to
        provide a simple *Up / Down* prediction. Use the side menu to explore the data (EDA), run single-stock
        predictions, or get a quick real-time check for today's stocks.
        """
    )

    st.header("Project Overview")
    st.markdown(
        "This Streamlit app loads a dataset and trained model automatically.\n\n"
        "Features included:\n"
        "- Interactive EDA with filterable stock list\n"
        "- Single-stock prediction using your saved model (Up / Down)\n"
        "- Real-time prediction using live data (yfinance)"
    )

    if df_main is not None:
        st.success(f"Dataset loaded: {LOCAL_DATA_FILE}")
    if model_path:
        st.success(f"Model loaded: {model_path}")


elif page == "EDA":
    st.title("Exploratory Data Analysis")

    if df_main is None:
        st.warning("No dataset loaded.")
    else:
        st.subheader("Dataset preview")
        st.write(f"Rows: {df_main.shape[0]} — Columns: {df_main.shape[1]}")
        st.dataframe(df_main.head(50))

        cols = df_main.columns.tolist()
        ticker_col = st.sidebar.selectbox("Ticker / Stock column", options=[None]+cols, index=0)
        date_col = st.sidebar.selectbox("Date column", options=[None]+cols, index=0)
        price_col = st.sidebar.selectbox("Price column", options=[None]+cols, index=0)

        selected_symbol = None
        if ticker_col:
            unique_vals = df_main[ticker_col].dropna().unique().tolist()
            selected_symbol = st.sidebar.selectbox("Pick a stock to inspect", options=[None]+unique_vals)

        if date_col and price_col:
            try:
                df_main[date_col] = pd.to_datetime(df_main[date_col])
                df_plot = df_main[df_main[ticker_col]==selected_symbol].sort_values(date_col) if selected_symbol else df_main.sort_values(date_col)

                st.subheader("Price over time")
                fig = px.line(df_plot, x=date_col, y=price_col, title=f"{price_col} over time")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Summary statistics")
                st.write(df_plot[price_col].describe())

            except Exception as e:
                st.error(f"Failed to generate visual: {e}")

elif page == "Prediction":
    st.title("Single-stock Prediction")

    if df_main is None:
        st.warning("No dataset loaded.")
    elif model_obj is None:
        st.warning("No model detected.")
    else:
        ticker_candidates = [c for c in df_main.columns if c.lower() in ("ticker","symbol","name","stock","code")]
        ticker_col = ticker_candidates[0] if ticker_candidates else st.selectbox("Ticker column", options=df_main.columns)
        symbols = sorted(df_main[ticker_col].dropna().unique().tolist())
        chosen = st.selectbox("Choose stock", options=symbols)

        df_sym = df_main[df_main[ticker_col] == chosen].copy()
        if not df_sym.empty:
            last_row = df_sym.sort_index().iloc[-1]
            st.write("Preview of data row used for prediction:")
            st.write(last_row)

            X = prepare_features_for_model(last_row)
            try:
                pred = model_obj.predict(X)
                label = str(pred[0])
                if label in [0, '0', 'down', 'Down', 'DOWN']:
                    st.warning(f"Model predicts: DOWN ({label})")
                else:
                    st.success(f"Model predicts: UP ({label})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Real-time Prediction":
    st.title("Real-time / Today's Prediction")

    if df_main is None or model_obj is None:
        st.warning("Dataset or model not loaded.")
    else:
        ticker_candidates = [c for c in df_main.columns if c.lower() in ("ticker","symbol","name","stock","code")]
        if ticker_candidates:
            ticker_col = ticker_candidates[0]
            symbols = sorted(df_main[ticker_col].dropna().unique().tolist())
            chosen = st.selectbox("Choose stock to check live", options=symbols)

            if st.button("Fetch latest and predict"):
                try:
                    yf_ticker = yf.Ticker(chosen)
                    hist = yf_ticker.history(period="5d", interval="1d")
                    if hist.empty:
                        st.error("No live data available for this ticker via yfinance.")
                    else:
                        latest_row = hist.tail(1).iloc[-1]
                        st.dataframe(hist.tail(1))

                        Xlive = prepare_features_for_model(latest_row)
                        pred = model_obj.predict(Xlive)
                        label = str(pred[0])
                        if label in [0, '0', 'down', 'Down', 'DOWN']:
                            st.warning(f"Model predicts: DOWN ({label})")
                        else:
                            st.success(f"Model predicts: UP ({label})")
                except Exception as e:
                    st.error(f"Failed to fetch live data: {e}")
        else:
            st.info("Could not detect a ticker column in your dataset.")

else:  # About
    st.title("About this App")
    st.markdown(
        """
        **Streamlit Stock Analysis & Prediction**\n
        This app provides:
        - EDA with interactive charts and filters
        - Model inference (Up/Down)
        - Quick real-time checks via yfinance

        **Notes**:
        - Replace `prepare_features_for_model` with your trained feature pipeline.
        - Dataset is auto-downloaded from Google Drive; no need to upload manually.
        """
    )
