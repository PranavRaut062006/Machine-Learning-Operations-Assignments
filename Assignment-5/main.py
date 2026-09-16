from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Adult Income Prediction API",
    description="REST API for Adult Income Prediction using Machine Learning",
    version="1.0.0"
)

# Load trained models
decision_tree_model = joblib.load("models/decision_tree_model.pkl")
random_forest_model = joblib.load("models/random_forest_model.pkl")

@app.get("/")
def home():
    return {
        "message": "Adult Income Prediction API is running",
        "status": "success"
    }

class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# def prepare_input(data: PredictionInput):
#     return [[
#         data.age,
#         data.workclass,
#         data.fnlwgt,
#         data.education,
#         data.education_num,
#         data.marital_status,
#         data.occupation,
#         data.relationship,
#         data.race,
#         data.sex,
#         data.capital_gain,
#         data.capital_loss,
#         data.hours_per_week,
#         data.native_country
#     ]]
def prepare_input(data: PredictionInput):
    return pd.DataFrame([data.model_dump()])

@app.post("/predict")
def predict(data: PredictionInput):
    input_data = prepare_input(data)

    dt_prediction = decision_tree_model.predict(input_data)[0]
    rf_prediction = random_forest_model.predict(input_data)[0]

    return {
        "decision_tree_prediction": dt_prediction,
        "random_forest_prediction": rf_prediction
    }


@app.get("/models")
def get_models():
    return {
        "models": [
            "Decision Tree",
            "Random Forest"
        ],
        "purpose": "Adult Income Prediction",
        "status": "loaded"
    }

predictions = {}
next_id = 1

@app.put("/predictions/{prediction_id}")
def update_prediction(prediction_id: int, data: PredictionInput):
    input_data = prepare_input(data)

    dt_prediction = decision_tree_model.predict(input_data)[0]
    rf_prediction = random_forest_model.predict(input_data)[0]

    predictions[prediction_id] = {
        "input": data.model_dump(),
        "decision_tree_prediction": dt_prediction,
        "random_forest_prediction": rf_prediction
    }

    return {
        "message": "Prediction completely updated",
        "prediction_id": prediction_id,
        "data": predictions[prediction_id]
    }

@app.patch("/predictions/{prediction_id}")
def patch_prediction(prediction_id: int, data: PredictionInput):
    if prediction_id not in predictions:
        return {
            "message": "Prediction ID not found"
        }

    input_data = prepare_input(data)

    dt_prediction = decision_tree_model.predict(input_data)[0]
    rf_prediction = random_forest_model.predict(input_data)[0]

    predictions[prediction_id].update({
        "input": data.model_dump(),
        "decision_tree_prediction": dt_prediction,
        "random_forest_prediction": rf_prediction
    })

    return {
        "message": "Prediction partially updated",
        "prediction_id": prediction_id,
        "data": predictions[prediction_id]
    }

@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int):
    if prediction_id not in predictions:
        return {
            "message": "Prediction ID not found"
        }

    del predictions[prediction_id]

    return {
        "message": "Prediction deleted successfully",
        "prediction_id": prediction_id
    }


@app.get("/predictions")
def get_predictions():
    return {
        "total_predictions": len(predictions),
        "predictions": predictions
    }