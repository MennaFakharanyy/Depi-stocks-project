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
                loaded = joblib.load(p)
                # Check if it's a valid model with predict method
                if hasattr(loaded, "predict"):
                    return loaded, p
                else:
                    st.warning(f"{p} loaded but is not a valid trained model.")
                    return None, None
            except Exception as e:
                st.warning(f"Failed to load {p}: {e}")
                continue
    return None, None

def prepare_features_for_model(row):
    """
    Converts a Series or DataFrame row into numeric ML-ready features.
    This ensures consistency for both historical dataset rows and
    live real-time price rows from yfinance.
    """
    import pandas as pd
    import numpy as np

    # If row is a Pandas Series (e.g., yfinance latest row), convert to DataFrame
    if isinstance(row, pd.Series):
        row = row.to_frame().T

    # Keep only numeric columns (the model ignores date/text fields)
    numeric_df = row.select_dtypes(include=[np.number]).fillna(0)

    # Convert to numpy array: shape (1, n_features)
    return numeric_df.values.reshape(1, -1)

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
    st.title("📈 Real-Time Stock Prediction & Live Price Check")

    if df_main is None:
        st.warning("Dataset not loaded.")
        st.stop()

    if model_obj is None or not hasattr(model_obj, "predict"):
        st.warning("No valid trained model loaded. Please upload a proper .pkl model.")
        st.stop()

    # ----------------------------
    # Detect ticker column
    # ----------------------------
    ticker_candidates = [c for c in df_main.columns if c.lower() in ("ticker","symbol","name","stock","code")]
    if not ticker_candidates:
        st.error("❌ No ticker/symbol column found in dataset.")
        st.stop()
    ticker_col = ticker_candidates[0]

    dataset_symbols = sorted(df_main[ticker_col].dropna().unique().tolist())
    st.subheader("Select Prediction Mode")
    mode = st.radio(
        "Prediction Source",
        ["From Dataset (Historical)", "Real-Time (YFinance)"],
        help="Choose whether to predict using your dataset or live market data."
    )

    # Features expected by the model
    model_features = ['volatility_20','volatility_50','SMA_20','SMA_50','RSI_14',
                      'day_of_week','is_month_end','is_quarter_end','daily_return']

    # =========================
    # 1️⃣ Historical Dataset
    # =========================
    if mode == "From Dataset (Historical)":
        st.info("Uses your cleaned dataset to show the last available record.")
        symbol = st.selectbox("Choose stock:", dataset_symbols)

        if st.button("Predict Using Dataset"):
            try:
                stock_rows = df_main[df_main[ticker_col]==symbol]
                if stock_rows.empty:
                    st.warning("No rows for this stock in your dataset.")
                    st.stop()

                latest_row = stock_rows.tail(1).iloc[-1]
                st.subheader("📊 Latest Data in Dataset")
                st.dataframe(stock_rows.tail(1))

                X = latest_row[model_features].values.reshape(1, -1)
                pred = model_obj.predict(X)[0]
                direction = "↗️ Up" if pred == 1 else "↘️ Down"

                st.success(f"Prediction for {symbol}: {direction}")
                st.write("Latest close:", float(latest_row['close']))

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

    # =========================
    # 2️⃣ Real-Time via YFinance
    # =========================
    elif mode == "Real-Time (YFinance)":
        st.info("Fetches live market prices from Yahoo Finance. Model prediction requires the symbol to exist in your dataset.")
        symbol = st.selectbox("Choose stock (must exist in dataset):", dataset_symbols)

        if st.button("Fetch Live Price & Predict"):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="6mo", interval="1d").reset_index()
                df.rename(columns={"Date":"date","Close":"close","Open":"open"}, inplace=True)

                # Compute features
                df['daily_return'] = df['close'].pct_change()
                for w in [20,50]:
                    df[f'volatility_{w}'] = df['daily_return'].rolling(w).std()
                df['SMA_20'] = df['close'].rolling(20).mean()
                df['SMA_50'] = df['close'].rolling(50).mean()

                # RSI 14
                delta = df['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(com=13, adjust=False).mean()
                avg_loss = loss.ewm(com=13, adjust=False).mean()
                rs = avg_gain / avg_loss
                df['RSI_14'] = 100 - (100 / (1 + rs))

                df['day_of_week'] = df['date'].dt.dayofweek
                df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
                df['is_quarter_end'] = (df['date'].dt.month % 3 == 0).astype(int)
                df = df.dropna()

                latest_row = df.iloc[-1]
                prev_close = df.iloc[-2]['close']

                st.subheader("📊 Live Market Data")
                st.write(f"**Symbol:** {symbol}")
                st.write(f"**Date:** {latest_row['date'].date()}")
                st.metric("Latest Close Price", f"${latest_row['close']:.2f}",
                          f"{latest_row['close'] - prev_close:.2f}")
                st.dataframe(df.tail(5))

                # Predict
                Xlive = latest_row[model_features].values.reshape(1, -1)
                pred = model_obj.predict(Xlive)[0]
                direction = "↗️ Up" if pred == 1 else "↘️ Down"

                st.success(f"Prediction for {symbol}: {direction}")
                st.write("Latest close:", float(latest_row['close']))

            except Exception as e:
                st.error(f"❌ Error fetching or predicting real-time data: {e}")
    
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
