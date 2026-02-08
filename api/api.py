import logging
import os

import joblib
import mlflow
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
    version="1.1.0",
)

# -------------------------------------------------------------------
# Model loading configuration
# -------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SPYDirectionModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

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
    """
    Loads model from MLflow Registry using alias.
    Falls back to local sklearn model if MLflow is unavailable.
    """
    if USE_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MODEL_ALIAS}"
            logger.info("Loading model from MLflow: %s", model_uri)

            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(
                "Model loaded successfully from MLflow",
                extra={
                    "model_name": MLFLOW_MODEL_NAME,
                    "alias": MODEL_ALIAS,
                },
            )
            return model

        except Exception:
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
    """
    Downloads recent market data and builds features identical
    to training-time feature engineering.
    """
    logger.info("Downloading market data from yfinance")

    # We need enough history to compute rolling features
    df = yf.download("SPY", period="30d", interval="1d", progress=False)

    df["return"] = df["Close"].pct_change()
    df["ma_3"] = df["Close"].rolling(3).mean()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["volatility_3"] = df["return"].rolling(3).std()

    df = df.dropna()

    latest = df.iloc[-1]

    feature_cols = ["return", "ma_3", "ma_5", "volatility_3"]

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

    # ----------------------------------------------------------------
    # MLflow-safe prediction handling
    # ----------------------------------------------------------------
    pred = model.predict(X)

    if isinstance(pred, pd.DataFrame):
        score = float(pred.iloc[0, 0])
    elif isinstance(pred, pd.Series):
        score = float(pred.iloc[0])
    elif isinstance(pred, np.ndarray):
        score = float(pred[0])
    else:
        score = float(pred)

    prediction = "UP" if score >= threshold else "DOWN"

    logger.info(
        "Prediction completed",
        extra={
            "score": score,
            "threshold": threshold,
            "prediction": prediction,
            "model_alias": MODEL_ALIAS,
            "model_source": "mlflow" if USE_MLFLOW else "local",
        },
    )

    return {
        "prediction": prediction,
        "confidence": round(score, 3),
        "threshold": threshold,
        "model_source": "mlflow" if USE_MLFLOW else "local",
        "model_alias": MODEL_ALIAS,
    }
