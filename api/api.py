import logging
import os

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ml-api")

# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI(
    title="Market Direction Prediction API",
    version="1.0.0",
)

# -------------------------------------------------------------------
# Model loading configuration
# -------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SPYDirectionModel")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "../train/model/model.joblib",
    ),
)

USE_MLFLOW = bool(MLFLOW_TRACKING_URI)

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
def load_model():
    if USE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MODEL_VERSION}"
            logger.info("Loading model from MLflow: %s", model_uri)

            model = mlflow.sklearn.load_model(model_uri)

            logger.info("Model loaded successfully from MLflow")
            return model

        except Exception as e:
            logger.exception(
                "Failed to load model from MLflow, falling back to local model"
            )

    logger.warning("Loading local model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Local model loaded successfully")
    return model


model = load_model()

# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# API endpoint
# -------------------------------------------------------------------
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

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # sklearn-style: shape (n_samples, n_classes)
        prob = float(proba.iloc[0, 1] if isinstance(proba, pd.DataFrame) else proba[0][1])

    else:
        pred = model.predict(X)

        # Handle pyfunc / sklearn / numpy scalar safely
        if isinstance(pred, pd.DataFrame):
            prob = float(pred.iloc[0, 0])
        elif isinstance(pred, pd.Series):
            prob = float(pred.iloc[0])
        elif isinstance(pred, np.ndarray):
            prob = float(pred[0])
        else:
            prob = float(pred)

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
        "model_source": "mlflow" if USE_MLFLOW else "local",
    }
