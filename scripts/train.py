import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load processed dataset
data = pd.read_csv("../data/processed/telecom_churn_processed.csv")

# Split features and target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/churn_model.pkl")
print("✅ Model saved to models/churn_model.pkl")
