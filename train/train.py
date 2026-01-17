import logging
import os
from contextlib import nullcontext
from datetime import timedelta

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from data_sources.yfinance_source import YFinanceDataSource
from mlflow.tracking import MlflowClient
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
EXPERIMENT_NAME = "spy-direction-training"
REGISTERED_MODEL_NAME = "SPYDirectionModel"

ANCHOR_OFFSET_DAYS = 365          # replay from 1 year ago
TRAIN_WINDOW_DAYS = 760           # ~2 years
PREDICTION_WINDOW_DAYS = 7        # 1-week prediction window

PROMOTION_THRESHOLD = 0.01

BASE_DIR = os.getenv("BASE_DIR", os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
USE_MLFLOW = bool(MLFLOW_TRACKING_URI)

if USE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow enabled (%s)", MLFLOW_TRACKING_URI)
else:
    logger.warning("MLflow disabled — local training only")

# -------------------------------------------------------------------
# Time anchor (HISTORICAL, NOT FUTURE)
# -------------------------------------------------------------------
ANCHOR_DATETIME = pd.Timestamp.utcnow() - timedelta(days=ANCHOR_OFFSET_DAYS)
train_end = ANCHOR_DATETIME
train_start = train_end - timedelta(days=TRAIN_WINDOW_DAYS)
eval_start = train_end
eval_end = train_end + timedelta(days=PREDICTION_WINDOW_DAYS)

logger.info("Anchor time: %s", ANCHOR_DATETIME.isoformat())

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
data_source = YFinanceDataSource(interval="1d")
logger.info("Loading market data for %s", TICKER)

df = data_source.load(
    TICKER,
    start=train_start,
    end=eval_end,
)

df.index = pd.to_datetime(df.index)
df = df.sort_index()

# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()

df = build_features(df)

# -------------------------------------------------------------------
# Train / eval split
# -------------------------------------------------------------------
train_df = df.loc[train_start:train_end]
eval_df = df.loc[eval_start:eval_end]

feature_cols = ["return", "ma_5", "ma_20", "volatility_10"]

X_train = train_df[feature_cols]
y_train = train_df["target"]

X_eval = eval_df[feature_cols]
y_eval = eval_df["target"]

logger.info(
    "Training rows: %d | Evaluation rows: %d",
    len(X_train),
    len(X_eval),
)

if len(X_eval) < 5:
    raise RuntimeError("Evaluation window too small — skipping run")

# -------------------------------------------------------------------
# Experiment grid
# -------------------------------------------------------------------
EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8},
]

# -------------------------------------------------------------------
# Calculate latest Production accuracy
# -------------------------------------------------------------------
def get_production_accuracy(model_name: str):
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get("accuracy")
    except Exception:
        return None

baseline_accuracy = get_production_accuracy(REGISTERED_MODEL_NAME)

if baseline_accuracy is None:
    logger.info("No Production model found — bootstrap mode")
else:
    logger.info("Baseline Production accuracy: %.4f", baseline_accuracy)

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
best_model = None
best_accuracy = -1.0
best_params = None

for params in EXPERIMENTS:
    with mlflow.start_run(nested=True) if USE_MLFLOW else nullcontext():
        logger.info("Experiment: %s", params)

        model = RandomForestClassifier(
            **params,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)

        logger.info("Eval accuracy: %.4f", acc)

        if USE_MLFLOW:
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.set_tag("anchor_time", ANCHOR_DATETIME.isoformat())
            mlflow.set_tag("train_window_days", TRAIN_WINDOW_DAYS)
            mlflow.set_tag("prediction_window_days", PREDICTION_WINDOW_DAYS)
            mlflow.set_tag("pipeline", "cronjob")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_params = params

# -------------------------------------------------------------------
# Promotion decision
# -------------------------------------------------------------------
should_promote = (
    baseline_accuracy is None
    or best_accuracy > baseline_accuracy + PROMOTION_THRESHOLD
)

logger.info(
    "Best accuracy: %.4f | Promote: %s",
    best_accuracy,
    should_promote,
)

# -------------------------------------------------------------------
# Register & promote
# -------------------------------------------------------------------
if USE_MLFLOW and should_promote:
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy)

        result = mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = MlflowClient()
        version = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=["None"]
        )[-1].version

        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "promotion_reason",
            "bootstrap" if baseline_accuracy is None else "better_than_baseline",
        )

        logger.info("Model promoted to Production (v%s)", version)

elif not USE_MLFLOW:
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Model saved locally to %s", MODEL_PATH)

logger.info("Training pipeline complete")
