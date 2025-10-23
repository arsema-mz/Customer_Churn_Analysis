import joblib
import pandas as pd

# Load saved model
model = joblib.load("../models/gradient_boosting_model.pkl")

# Example input data
sample_data = pd.DataFrame({
    'tenure': [24],
    'InternetService_Fiber optic': [1],
    'OnlineSecurity_No internet service': [0],
    'OnlineBackup_No internet service': [0],
    'DeviceProtection_No internet service': [1],
    'TechSupport_No internet service': [0],
    'StreamingTV_No internet service': [0],
    'StreamingMovies_No internet service': [1],
    'Contract_Two year': [1],
    'PaymentMethod_Electronic check': [0]
})

# Predict churn (1 = churn, 0 = no churn)
prediction = model.predict(sample_data)
print("🧠 Predicted Churn:", "Yes" if prediction[0] == 1 else "No")
