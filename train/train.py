import os

import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Settings
TICKER = "SPY"
BASE_DIR = os.getenv("BASE_DIR", os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.getenv(
    "MODEL_PATH",  # env variable in Docker/K8s
    os.path.join(MODEL_DIR, "model.joblib"),  # local dev
)

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Download historical data
df = yf.download(TICKER, period="5y", interval="1d")

# 2. Feature engineering
df["return"] = df["Close"].pct_change()
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["volatility_10"] = df["return"].rolling(10).std()

# 3. Target variable: will price go up tomorrow?
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# 4. Drop NA
df = df.dropna()

# 5. Features & labels
feature_cols = ["return", "ma_5", "ma_20", "volatility_10"]
X = df[feature_cols]
y = df["target"]

# 6. Train/test split (time-based)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 7. Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.4f}")

# 9. Save model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("Training complete.")
