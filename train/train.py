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
CHAMPION_ALIAS = "champion"

ANCHOR_OFFSET_DAYS = 365
TRAIN_WINDOW_DAYS = 365
PREDICTION_WINDOW_DAYS = 30
MIN_EVAL_SAMPLES = 7

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

client = MlflowClient() if USE_MLFLOW else None

# -------------------------------------------------------------------
# Time anchor (historical replay)
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
df = data_source.load(TICKER, start=train_start, end=eval_end)

df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
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

X_train, y_train = train_df[feature_cols], train_df["target"]
X_eval, y_eval = eval_df[feature_cols], eval_df["target"]

if len(X_eval) < MIN_EVAL_SAMPLES:
    logger.warning("Evaluation window too small, skipping run")
    exit(0)

# -------------------------------------------------------------------
# Experiment grid
# -------------------------------------------------------------------
EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8},
]

# -------------------------------------------------------------------
# Get baseline (champion alias)
# -------------------------------------------------------------------
def get_champion_accuracy(model_name: str):
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        run = client.get_run(mv.run_id)
        return run.data.metrics.get("accuracy")
    except Exception:
        return None

baseline_accuracy = get_champion_accuracy(REGISTERED_MODEL_NAME)

if baseline_accuracy is None:
    logger.info("No champion model found — bootstrap mode")
else:
    logger.info("Baseline champion accuracy: %.4f", baseline_accuracy)

# -------------------------------------------------------------------
# Training & Offline Evaluation loop
# -------------------------------------------------------------------
best_model, best_accuracy, best_params = None, -1.0, None

for params in EXPERIMENTS:
    with mlflow.start_run(run_name=f"rf_{params}") if USE_MLFLOW else nullcontext():
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_eval, model.predict(X_eval))
        logger.info("Params=%s | Accuracy=%.4f", params, acc)

        if USE_MLFLOW:
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.set_tags({
                "anchor_time": ANCHOR_DATETIME.isoformat(),
                "train_window_days": TRAIN_WINDOW_DAYS,
                "prediction_window_days": PREDICTION_WINDOW_DAYS,
                "evaluation_type": "offline",
                "pipeline": "cronjob",
            })

        if acc > best_accuracy:
            best_model, best_accuracy, best_params = model, acc, params

# -------------------------------------------------------------------
# Promotion decision
# -------------------------------------------------------------------
should_promote = (
    baseline_accuracy is None
    or best_accuracy > baseline_accuracy + PROMOTION_THRESHOLD
)

logger.info("Best accuracy=%.4f | Promote=%s", best_accuracy, should_promote)

# -------------------------------------------------------------------
# Register & alias promotion
# -------------------------------------------------------------------
if USE_MLFLOW and should_promote:
    with mlflow.start_run(run_name=f"PROMOTION_{ANCHOR_DATETIME:%Y%m%d_%H%M}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy)

        result = mlflow.sklearn.log_model(
            best_model,
            artifact_path="random_forest_model-yfinance",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        version = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )[0].version

        # Tag the model version (environment & promotion reason)
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "env",
            "dev"
        )

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "promotion_reason",
            "bootstrap" if baseline_accuracy is None else "better_than_champion"
        )

        # Set alias (routing only)
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
            version,
        )

        logger.info("Promoted model v%s to alias '%s'", version, CHAMPION_ALIAS)

elif not USE_MLFLOW:
    joblib.dump(best_model, MODEL_PATH)
    logger.info("Model saved locally to %s", MODEL_PATH)

logger.info("Training pipeline complete")
