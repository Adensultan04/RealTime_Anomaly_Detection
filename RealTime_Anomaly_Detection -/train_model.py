import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("transactions.csv")

# Select numeric features only
features = df[["amount"]]

# Scale data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(scaled_features)

# Save model and scaler
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")