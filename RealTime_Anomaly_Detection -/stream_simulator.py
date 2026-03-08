import pandas as pd
import joblib
import time

# Load model and scaler
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("transactions.csv")

print("Starting Real-Time Simulation...\n")

for index, row in df.iterrows():
    
    amount = row["amount"]
    
    # Prepare data
    scaled_amount = scaler.transform([[amount]])
    
    prediction = model.predict(scaled_amount)
    
    if prediction[0] == -1:
        print(f"⚠ ANOMALY DETECTED! Transaction ID: {row['transaction_id']} | Amount: {amount}")
    else:
        print(f"Normal Transaction | ID: {row['transaction_id']} | Amount: {amount}")
    
    time.sleep(0.5)  # simulate real-time delay