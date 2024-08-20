from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi.responses import JSONResponse, StreamingResponse
import matplotlib.pyplot as plt
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI LSTM Pre-trained Modeling Prediction Application for Accidents Number!"}

class PredictionRequest(BaseModel):
    year: int
    month: int

# Load the pre-trained model and scaler
model = load_model('pretrained_lstm_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.load('scaler.npy'))

# Load your data
df = pd.read_csv('sample_data.csv')
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df.set_index('Date', inplace=True)
df = df[['Value']]
scaled_data = scaler.transform(df)

seq_length = 12

# Helper function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Function to predict future values
def predict_future(sequence, steps, model, scaler):
    predictions = []
    current_sequence = sequence
    for _ in range(steps):
        prediction = model.predict(current_sequence[:, :-1])
        prediction = scaler.inverse_transform(prediction)
        predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[:, 1:], [[prediction]], axis=1)
    return predictions

@app.post("/predict/")
async def predict(input_data: PredictionRequest):
    try:
        target_date = datetime(input_data.year, input_data.month, 1)
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, 1))
        predicted_value = model.predict(last_sequence[:, :-1])
        predicted_value = scaler.inverse_transform(predicted_value)
        
        forecast_json = {
            "prediction": float(predicted_value[0][0])
        }
        return JSONResponse(content=forecast_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/errors/")
async def errors(input_data: PredictionRequest):
    try:
        # Create sequences for the entire dataset
        sequences = create_sequences(scaled_data, seq_length)
        X = sequences[:, :-1]
        y = sequences[:, -1]
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Predict on the entire dataset
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        errors_list = []
        test_dates = df.index[seq_length:]  # Get the corresponding dates for the sequences
        
        for date, actual, predicted in zip(test_dates, y_test, y_pred):
            errors_list.append({
                "year": date.year,
                "month": date.month,
                "actual_value": float(actual[0]),
                "predicted_value": float(predicted[0]),
                "error": float(abs(actual[0] - predicted[0]))
            })
        
        error_json = {
            "mae": mae,
            "mse": mse,
            "details": errors_list
        }
        return JSONResponse(content=error_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
