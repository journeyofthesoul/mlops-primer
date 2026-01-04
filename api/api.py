import os

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI


app = FastAPI()

MODEL_PATH = os.getenv(
    "MODEL_PATH",  # env variable in Docker/K8s
    os.path.join(
        os.path.dirname(__file__),
        "../train/model/model.joblib",
    ),  # local dev
)

# Load model at startup
model = joblib.load(MODEL_PATH)


def get_latest_features():
    df = yf.download("SPY", period="30d", interval="1d")
    df["return"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    df = df.dropna()
    latest = df.iloc[-1]
    feature_cols = ["return", "ma_5", "ma_20", "volatility_10"]
    return pd.DataFrame([latest[feature_cols]])


@app.get("/predictionForTomorrow")
def predict():
    X = get_latest_features()
    prob = model.predict_proba(X)[0][1]
    return {
        "prediction": "UP" if prob > 0.5 else "DOWN",
        "confidence": round(float(prob), 3),
    }
