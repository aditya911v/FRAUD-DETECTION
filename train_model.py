import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Step 1: Load the dataset
data = pd.read_csv("fraud_domains.csv")  # Make sure the dataset is in the same directory

# ✅ Step 2: Separate features and labels
X = data.drop(columns=["domain"])  # Drop domain name (not a feature)
y = data["is_fraudulent"]  # Labels (1 = fraud, 0 = safe)

# ✅ Step 3: Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 5: Save the trained model
joblib.dump(model, "fraud_detector.pkl")
print("🎉 AI Model Trained & Saved as fraud_detector.pkl!")
