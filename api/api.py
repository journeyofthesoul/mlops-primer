import logging
import os

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ml-api")


app = FastAPI(
    title="Market Direction Prediction API",
    version="1.0.0",
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",  # env variable in Docker/K8s
    os.path.join(
        os.path.dirname(__file__),
        "../train/model/model.joblib",
    ),  # local dev
)

logger.info("Loading model from %s", MODEL_PATH)
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully")


def get_latest_features() -> pd.DataFrame:
    logger.info("Downloading market data from yfinance")

    df = yf.download("SPY", period="30d", interval="1d", progress=False)

    df["return"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    df = df.dropna()

    latest = df.iloc[-1]
    feature_cols = ["return", "ma_5", "ma_20", "volatility_10"]

    logger.debug("Latest features: %s", latest[feature_cols].to_dict())

    return pd.DataFrame([latest[feature_cols]])


@app.get(
    "/predictionForTomorrow",
    summary="Predict market direction for tomorrow",
)
def predict(
    threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for UP/DOWN classification",
    ),
):
    logger.info("Prediction request received (threshold=%.3f)", threshold)

    X = get_latest_features()
    prob = float(model.predict_proba(X)[0][1])

    prediction = "UP" if prob > threshold else "DOWN"

    logger.info(
        "Prediction completed",
        extra={
            "probability": prob,
            "threshold": threshold,
            "prediction": prediction,
        },
    )

    return {
        "prediction": prediction,
        "confidence": round(prob, 3),
        "threshold": threshold,
    }
