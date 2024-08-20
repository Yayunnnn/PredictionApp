
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from fastapi.responses import JSONResponse, StreamingResponse
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI LSTM Modeling Predcition Application for Accidents Number!"}

class PredictionRequest(BaseModel):
    year: int
    month: int

# Load your data
df = pd.read_csv('sample_data.csv')
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df.set_index('Date', inplace=True)
df = df[['Value']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 12
sequences = create_sequences(scaled_data, seq_length)
X = sequences[:, :-1]
y = sequences[:, -1]

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length - 1, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

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
        target_date = datetime(input_data.year, input_data.month, 1)
        # Generate the prediction for the target date
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, 1))
        predicted_value = model.predict(last_sequence[:, :-1])
        predicted_value = scaler.inverse_transform(predicted_value)
        
        # Compute error metrics
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test_inverse, y_pred)
        mse = mean_squared_error(y_test_inverse, y_pred)
        
        # Prepare detailed error information
        errors_list = []
        test_dates = df.index[-len(y_test_inverse):]  # Get the corresponding dates for the test set
        
        for date, actual, predicted in zip(test_dates, y_test_inverse, y_pred):
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

@app.post("/plot/")
async def plot(input_data: PredictionRequest):
    try:
        target_date = datetime(input_data.year, input_data.month, 1)
        last_sequence = scaled_data[-seq_length:]
        last_sequence = last_sequence.reshape((1, seq_length, 1))
        predicted_value = model.predict(last_sequence[:, :-1])
        predicted_value = scaler.inverse_transform(predicted_value)
        
        # Extend the historical data with the predicted value
        extended_dates = df.index.tolist() + [target_date]
        extended_values = df['Value'].tolist() + [predicted_value[0][0]]
        
        # Plot historical data and predicted value
        plt.figure(figsize=(10, 6))
        plt.plot(extended_dates, extended_values, label='Historical Data with Prediction', color='b')
        plt.axvline(x=target_date, color='r', linestyle='--', label='Prediction Date')
        plt.scatter([target_date], [predicted_value[0][0]], color='r', label='Predicted Value')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Historical Data and Predicted Value')
        plt.legend()
        
        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))