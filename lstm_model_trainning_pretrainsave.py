import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv('sample_data.csv')
# fielter data before 2021
df = df[df['Year'] < 2021]
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

# Build the LSTM model for pre-training
model = Sequential()

# Use Bidirectional LSTM for better performance in sequence prediction
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_length - 1, 1))))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(25, activation='relu'))  # Fully connected layer
model.add(Dense(1))  # Output layer

# Compile the model with an adaptive learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Early stopping to avoid overfitting and save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('pretrained_lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')

# Pre-train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, checkpoint])

# Save the scaler for later use
np.save('scaler.npy', scaler.fit_transform(df))

# Evaluate the model on test data
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_inverse, y_pred)
mse = mean_squared_error(y_test_inverse, y_pred)

print(f"Pre-training MAE: {mae}")
print(f"Pre-training MSE: {mse}")

# The model is now pre-trained and saved as 'pretrained_lstm_model.h5'

# Load the pre-trained model
model = load_model('pretrained_lstm_model.h5')

# Fine-tune the model on the target dataset (if different from pre-training data)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, checkpoint])

# Evaluate the fine-tuned model
y_pred_fine_tuned = model.predict(X_test)
y_pred_fine_tuned = scaler.inverse_transform(y_pred_fine_tuned)

mae_fine_tuned = mean_absolute_error(y_test_inverse, y_pred_fine_tuned)
mse_fine_tuned = mean_squared_error(y_test_inverse, y_pred_fine_tuned)

print(f"Fine-tuned MAE: {mae_fine_tuned}")
print(f"Fine-tuned MSE: {mse_fine_tuned}")


