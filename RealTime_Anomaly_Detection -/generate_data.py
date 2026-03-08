import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

num_transactions = 1000

data = []

start_time = datetime.now()

for i in range(num_transactions):
    transaction_id = i + 1
    customer_id = random.randint(1, 50)
    
    # Normal transaction amount
    amount = np.random.normal(200, 50)
    
    # Inject anomaly (2% high-value fraud)
    if random.random() < 0.02:
        amount = np.random.uniform(5000, 15000)
    
    timestamp = start_time + timedelta(seconds=i*10)
    
    location = random.choice(["Karachi", "Lahore", "Islamabad"])
    payment_method = random.choice(["Card", "Cash", "Online"])
    
    data.append([transaction_id, customer_id, amount, timestamp, location, payment_method])

df = pd.DataFrame(data, columns=[
    "transaction_id",
    "customer_id",
    "amount",
    "timestamp",
    "location",
    "payment_method"
])

df.to_csv("transactions.csv", index=False)

print("Dataset generated successfully!")