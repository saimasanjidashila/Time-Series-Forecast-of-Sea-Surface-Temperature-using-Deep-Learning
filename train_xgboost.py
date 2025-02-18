#!/usr/bin/env python3

#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt

# **Function to calculate metrics**
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

# **Step 1: Load the pre-split datasets**
train_data = pd.read_csv('./dataset/train_dataset.csv')
val_data = pd.read_csv('./dataset/validation_dataset.csv')
test_data = pd.read_csv('./dataset/test_dataset.csv')

train_data_dust = pd.read_csv('./dust/Dust_data/train_dataset.csv')
val_data_dust = pd.read_csv('./dust/Dust_data/validation_dataset.csv')
test_data_dust = pd.read_csv('./dust/Dust_data/test_dataset.csv')

# Convert time column to datetime
for df in [train_data, val_data, test_data, train_data_dust, val_data_dust, test_data_dust]:
    df['time'] = pd.to_datetime(df['time'])

# **Step 2: Extract year, month, and day from the time column**
for df in [train_data, val_data, test_data, train_data_dust, val_data_dust, test_data_dust]:
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day

# **Step 3: Define features and target**
features = ['year', 'month', 'day', 'lat', 'lon']
features_dust = ['year', 'month', 'day', 'lat', 'lon', 'DUCMASS']

X_train, y_train = train_data[features], train_data['sst']
X_val, y_val = val_data[features], val_data['sst']
X_test, y_test = test_data[features], test_data['sst']

X_train_dust, y_train_dust = train_data_dust[features_dust], train_data_dust['sst']
X_val_dust, y_val_dust = val_data_dust[features_dust], val_data_dust['sst']
X_test_dust, y_test_dust = test_data_dust[features_dust], test_data_dust['sst']

print("✅ Data processing complete.")

# **Step 4: Initialize Models**
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5, n_estimators=200, random_state=42)
xgb_model_dust = xgb.XGBRegressor(objective='reg:squarederror', max_depth=5, n_estimators=50, random_state=42)

# **Step 5: Create a folder to save models**
save_dir = './xgboost_model'
os.makedirs(save_dir, exist_ok=True)

# **Train and evaluate XGBoost without dust**
print("\n▶️ Training XGBoost without dust...")
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# **Train and evaluate XGBoost with dust**
print("\n▶️ Training XGBoost with dust...")
xgb_model_dust.fit(X_train_dust, y_train_dust)
xgb_val_pred_dust = xgb_model_dust.predict(X_val_dust)
xgb_test_pred_dust = xgb_model_dust.predict(X_test_dust)

# **Calculate metrics**
xgb_val_mse, xgb_val_mae, xgb_val_rmse = calculate_metrics(y_val, xgb_val_pred)
xgb_test_mse, xgb_test_mae, xgb_test_rmse = calculate_metrics(y_test, xgb_test_pred)

xgb_val_mse_dust, xgb_val_mae_dust, xgb_val_rmse_dust = calculate_metrics(y_val_dust, xgb_val_pred_dust)
xgb_test_mse_dust, xgb_test_mae_dust, xgb_test_rmse_dust = calculate_metrics(y_test_dust, xgb_test_pred_dust)

print(f"✅ XGBoost (No Dust) - Validation: MSE={xgb_val_mse:.2f}, MAE={xgb_val_mae:.2f}, RMSE={xgb_val_rmse:.2f} | "
      f"Test: MSE={xgb_test_mse:.2f}, MAE={xgb_test_mae:.2f}, RMSE={xgb_test_rmse:.2f}")

print(f"✅ XGBoost (With Dust) - Validation: MSE={xgb_val_mse_dust:.2f}, MAE={xgb_val_mae_dust:.2f}, RMSE={xgb_val_rmse_dust:.2f} | "
      f"Test: MSE={xgb_test_mse_dust:.2f}, MAE={xgb_test_mae_dust:.2f}, RMSE={xgb_test_rmse_dust:.2f}")

# **Save XGBoost Models**
joblib.dump(xgb_model, os.path.join(save_dir, 'xgboost_sst_model.pkl'))
joblib.dump(xgb_model_dust, os.path.join(save_dir, 'xgboost_sst_dust_model.pkl'))
print(f"💾 XGBoost models saved in '{save_dir}' folder!")

# **Plot Actual vs Predicted SST for Test Data**
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label='Actual SST', color='black')
plt.plot(xgb_test_pred[:200], label='Predicted SST', color='blue')
plt.plot(xgb_test_pred_dust[:200], label='Predicted SST with Dust)', color='red')
plt.xlabel('Time, Lat, Lon')
plt.ylabel('Sea Surface Temperature (SST)')
plt.title('Actual SST vs Predicted SST (With and Without Dust)')
plt.legend()
plt.show()
