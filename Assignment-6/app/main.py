from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# Load trained model
MODEL_PATH = "models/churn_model.pkl"
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting Telco customer churn",
    version="1.0.0"
)


# Input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH)
    }


# Model information endpoint
@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest Classifier",
        "dataset": "Telco Customer Churn",
        "target": "Churn",
        "accuracy": 0.7779,
        "version": "1.0"
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    churn_prediction = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(probability), 4)
    }