#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import datetime

# 🔚 Function to Calculate Metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

# 🔚 Step 1: Load the pre-split datasets
train_data = pd.read_csv('./dataset/train_dataset.csv')
val_data = pd.read_csv('./dataset/validation_dataset.csv')
test_data = pd.read_csv('./dataset/test_dataset.csv')

train_data_dust = pd.read_csv('./dust/Dust_data/train_dataset.csv')
val_data_dust = pd.read_csv('./dust/Dust_data/validation_dataset.csv')
test_data_dust = pd.read_csv('./dust/Dust_data/test_dataset.csv')

# Convert time column to datetime and extract features
for df in [train_data, val_data, test_data, train_data_dust, val_data_dust, test_data_dust]:
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day

# 🔚 Step 2: Define Features and Target
features = ['year', 'month', 'day', 'lat', 'lon']
features_dust = ['year', 'month', 'day', 'lat', 'lon', 'DUCMASS']

# Data without dust
X_train, y_train = train_data[features], train_data[['sst']]
X_val, y_val = val_data[features], val_data[['sst']]
X_test, y_test = test_data[features], test_data[['sst']]

# Data with dust
X_train_dust, y_train_dust = train_data_dust[features_dust], train_data_dust[['sst']]
X_val_dust, y_val_dust = val_data_dust[features_dust], val_data_dust[['sst']]
X_test_dust, y_test_dust = test_data_dust[features_dust], test_data_dust[['sst']]

print('✅ Data processing done...')

# 🔚 Step 3: Normalize the Features and Target
scaler_X = MinMaxScaler()
scaler_X_dust = MinMaxScaler()
scaler_y = MinMaxScaler()

# Without dust
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# With dust
X_train_dust_scaled = scaler_X_dust.fit_transform(X_train_dust)
X_val_dust_scaled = scaler_X_dust.transform(X_val_dust)
X_test_dust_scaled = scaler_X_dust.transform(X_test_dust)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

y_train_dust_scaled = scaler_y.transform(y_train_dust)
y_val_dust_scaled = scaler_y.transform(y_val_dust)
y_test_dust_scaled = scaler_y.transform(y_test_dust)

# Save the scalers
os.makedirs('./CNN', exist_ok=True)
joblib.dump(scaler_X, './CNN/CNN_scaler_X.pkl')
joblib.dump(scaler_X_dust, './CNN/CNN_scaler_X_dust.pkl')
joblib.dump(scaler_y, './CNN/CNN_scaler_y.pkl')
print('✅ Scalers saved as CNN_scaler_X.pkl, CNN_scaler_X_dust.pkl, and CNN_scaler_y.pkl')

# Reshape for CNN input
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

X_train_dust_scaled = X_train_dust_scaled.reshape((X_train_dust_scaled.shape[0], X_train_dust_scaled.shape[1], 1))
X_val_dust_scaled = X_val_dust_scaled.reshape((X_val_dust_scaled.shape[0], X_val_dust_scaled.shape[1], 1))
X_test_dust_scaled = X_test_dust_scaled.reshape((X_test_dust_scaled.shape[0], X_test_dust_scaled.shape[1], 1))

# 🔚 Step 4: Build the CNN Model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Dropout(0.3),

        Conv1D(filters=64, kernel_size=2, activation='relu'),
        Dropout(0.3),

        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.4),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train without dust
model_no_dust = build_cnn_model((X_train_scaled.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

history_no_dust = model_no_dust.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=5,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Train with dust
model_with_dust = build_cnn_model((X_train_dust_scaled.shape[1], 1))
history_with_dust = model_with_dust.fit(
    X_train_dust_scaled, y_train_dust_scaled,
    validation_data=(X_val_dust_scaled, y_val_dust_scaled),
    epochs=5,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Save models
model_no_dust.save(f'./CNN/cnn_sst_model_no_dust.h5')
model_with_dust.save(f'./CNN/cnn_sst_model_with_dust.h5')
print("\n✅ Models saved!")

# Predictions and evaluations
val_pred_no_dust = scaler_y.inverse_transform(model_no_dust.predict(X_val_scaled))
val_pred_with_dust = scaler_y.inverse_transform(model_with_dust.predict(X_val_dust_scaled))

y_val_original = scaler_y.inverse_transform(y_val_scaled)

# Metrics
mse_no_dust, mae_no_dust, rmse_no_dust = calculate_metrics(y_val_original, val_pred_no_dust)
mse_with_dust, mae_with_dust, rmse_with_dust = calculate_metrics(y_val_original, val_pred_with_dust)

print(f"\n📊 Validation Set Metrics (No Dust):\nMSE: {mse_no_dust:.2f}, MAE: {mae_no_dust:.2f}, RMSE: {rmse_no_dust:.2f}")
print(f"\n📊 Validation Set Metrics (With Dust):\nMSE: {mse_with_dust:.2f}, MAE: {mae_with_dust:.2f}, RMSE: {rmse_with_dust:.2f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_val_original[:200], label='Actual SST', color='black')
plt.plot(val_pred_no_dust[:200], label='Predicted SST', color='blue')
plt.plot(val_pred_with_dust[:200], label='Predicted SST with Dust', color='red')
plt.title('Actual vs Predicted SST (First 200 Data Points)')
plt.xlabel('Time, Latitude, Longitude')
plt.ylabel('Sea Surface Temperature (SST)')
plt.legend()
plt.show()
