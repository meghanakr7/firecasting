# create a ML model for wildfire emission forecasting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

from fc_train_data_preprocess import read_original_txt_files

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")


# Assuming 'target' is the column to predict
target_col = ' FRP'

print("read the txt files into python dataframes")
df = read_original_txt_files()
# Lag/Shift the data for previous days' information
num_previous_days = 7  # Adjust the number of previous days to consider
for i in range(1, num_previous_days + 1):
    for col in df.columns:
        df[f'{col}_lag_{i}'] = df[col].shift(i)

# Drop rows with NaN values from the shifted columns
df.dropna(inplace=True)

# Define features and target
print("current all columns: ", df.columns)
X = df.drop([target_col], axis=1)
y = df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features - not sure about this
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train a model (e.g., Linear Regression)
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


