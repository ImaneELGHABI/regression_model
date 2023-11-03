from fastapi import FastAPI

from models import PredictionRequest
from regression_model.predict import make_prediction

app = FastAPI()

@app.get("/")
async def read_root():
    return {"response": "To make a prediction, send a POST request to /predict"}

# Predict endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a price prediction using a trained model."""
    # Extract data from request body
    input_data = request.input_data
    # Make prediction
    return make_prediction(input_data=input_data)
