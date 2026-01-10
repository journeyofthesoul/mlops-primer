import logging
import os

import joblib
import mlflow
import mlflow.sklearn
from data_sources.yfinance_source import YFinanceDataSource
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TICKER = "SPY"

BASE_DIR = os.getenv("BASE_DIR", os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(MODEL_DIR, "./model/model.joblib"),
)

os.makedirs(MODEL_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
USE_MLFLOW = bool(MLFLOW_TRACKING_URI)

if USE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("spy-direction-training")
    logger.info("MLflow enabled (tracking URI: %s)", MLFLOW_TRACKING_URI)
else:
    logger.warning("MLflow disabled — training will run locally only")

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
data_source = YFinanceDataSource()
logger.info("Loading market data for %s", TICKER)
df = data_source.load(TICKER)

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
logger.info("Performing feature engineering")

df["return"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["volatility_10"] = df["return"].rolling(10).std()
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

feature_cols = ["return", "ma_5", "ma_20", "volatility_10"]
X = df[feature_cols]
y = df["target"]

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------------------------------------------
# Training (with optional MLflow)
# -------------------------------------------------------------------
def train():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logger.info("Test accuracy: %.4f", acc)

    return model, acc


if USE_MLFLOW:
    with mlflow.start_run():
        mlflow.log_params(
            {
                "ticker": TICKER,
                "n_estimators": 100,
                "max_depth": 5,
                "train_split": 0.8,
            }
        )

        model, acc = train()

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SPYDirectionModel",
        )

        logger.info("Model logged to MLflow")
else:
    model, acc = train()

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved locally to %s", MODEL_PATH)

logger.info("Training complete")
