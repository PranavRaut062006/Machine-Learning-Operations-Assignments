import time
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response


# Load trained model
model = joblib.load("models/churn_model.pkl")

# Create FastAPI application
app = FastAPI(
   title="Telco Customer Churn Prediction API",
   description="ML model monitoring using Prometheus",
   version="1.0"
)


# -----------------------------
# Prometheus Metrics
# -----------------------------

REQUEST_COUNT = Counter(
   "prediction_requests_total",
   "Total number of prediction requests"
)

PREDICTION_COUNT = Counter(
   "churn_predictions_total",
   "Total churn predictions",
   ["prediction"]
)

REQUEST_LATENCY = Histogram(
   "prediction_latency_seconds",
   "Prediction request latency"
)

CHURN_PROBABILITY = Gauge(
   "churn_probability",
   "Latest predicted churn probability"
)


# -----------------------------
# Input Schema
# -----------------------------

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


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def home():
   return {
      "message": "Telco Customer Churn Prediction API",
      "status": "running"
   }


@app.post("/predict")
def predict(customer: CustomerData):

   start_time = time.time()

   REQUEST_COUNT.inc()

   input_data = customer.model_dump()

   input_df = pd.DataFrame([input_data])

   prediction = model.predict(input_df)[0]

   probability = model.predict_proba(input_df)[0][1]

   prediction_label = "Yes" if prediction == 1 else "No"

   PREDICTION_COUNT.labels(
      prediction=prediction_label
   ).inc()

   CHURN_PROBABILITY.set(probability)

   REQUEST_LATENCY.observe(
      time.time() - start_time
   )

   return {
      "churn_prediction": prediction_label,
      "churn_probability": round(float(probability), 4)
   }


@app.get("/metrics")
def metrics():
   return Response(
      content=generate_latest(),
      media_type="text/plain"
   )