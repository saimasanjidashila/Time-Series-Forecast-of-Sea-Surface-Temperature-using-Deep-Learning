#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model
model = load_model('./CNN/cnn_sst_model_with_dust.h5')

# Load test dataset (ensure the CSV file contains 'time', 'lat', 'lon')
test_data = pd.read_csv('./dust/Dust_data/test_dataset.csv')

# Convert time column to datetime and extract features
test_data['time'] = pd.to_datetime(test_data['time'])
test_data['year'] = test_data['time'].dt.year
test_data['month'] = test_data['time'].dt.month
test_data['day'] = test_data['time'].dt.day

# Select features used in training
feature_columns = ['year', 'month', 'day', 'lat', 'lon', 'DUCMASS']
X_test = test_data[feature_columns]

# Load the scaler for input features (features scaler saved during training)
scaler_X = joblib.load('./CNN/CNN_scaler_X_dust.pkl')

# Normalize test features using the same scaler from training
X_test_normalized = scaler_X.transform(X_test)

# Make predictions
predictions_normalized = model.predict(X_test_normalized)

# Load the scaler for SST (target variable)
scaler_y = joblib.load('./CNN/CNN_scaler_y.pkl')

# Inverse transform the predictions to get actual SST values
predictions = scaler_y.inverse_transform(predictions_normalized)

# Add the predicted SST values to the test dataset
test_data['Pred_SST'] = predictions.flatten()

# Combine year, month, and day into a single datetime column
test_data['date'] = pd.to_datetime(test_data[['year', 'month', 'day']])

# Select relevant columns for output
output_data = test_data[['date', 'lat', 'lon', 'DUCMASS','Pred_SST']]

# Save the predicted results to CSV
output_csv_path = './dust/Dust_data/predicted_sst.csv'
output_data.to_csv(output_csv_path, index=False, date_format='%Y-%m-%d')

print(f"Predictions saved to {output_csv_path}")
