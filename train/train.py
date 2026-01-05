import logging
import os

import joblib
from data_sources.yfinance_source import YFinanceDataSource
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


TICKER = "SPY"
BASE_DIR = os.getenv("BASE_DIR", os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(MODEL_DIR, "model.joblib"),
)

os.makedirs(MODEL_DIR, exist_ok=True)


data_source = YFinanceDataSource()
logger.info("Loading market data for %s", TICKER)
df = data_source.load(TICKER)


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


logger.info("Training model")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)


preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
logger.info("Test accuracy: %.4f", acc)


joblib.dump(model, MODEL_PATH)
logger.info("Model saved to %s", MODEL_PATH)
logger.info("Training complete")
