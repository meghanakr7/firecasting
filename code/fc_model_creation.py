# create a ML model for wildfire emission forecasting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_log_error, median_absolute_error, max_error
import xgboost as xgb
from lightgbm import LGBMRegressor
import warnings
import shutil
from datetime import datetime, timedelta
from fc_train_data_preprocess import training_data_folder

import pickle

from fc_train_data_preprocess import  prepare_training_data

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# final model path
#model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_latest.pkl"
#model_path = "/groups/ESS3/zsun/firecasting/model/fc_lightgbm_model_v2_latest.pkl"
model_path = "/groups/ESS3/zsun/firecasting/model/fc_lightgbm_model_v2_latest.pkl" # added the land use to distinguish the different geographical areas



def train_model(start_date_str, end_date_str, training_data_folder=training_data_folder):
  # start date and end date define the training period
  # Convert the date strings to datetime objects
  start_date = datetime.strptime(start_date_str, "%Y%m%d")
  end_date = datetime.strptime(end_date_str, "%Y%m%d")

  # Initialize a list to store the dates
  dates_between = []
  
  # Introduce a threshold for target values
  threshold = 100
  
  # Initialize and train a model (e.g., Linear Regression)
#   model = XGBRegressor(n_estimators=100,
#                        max_depth=8,
#                        learning_rate=0.1,)
  model = LGBMRegressor(n_jobs=-1, random_state=42)
                       #scale_pos_weight=abs((len(y_train) - y_train.sum()) / y_train.sum()))

  # Iterate through the days between start and end dates
  current_date = start_date
  label = 0
  target_col = 'FRP'
  
  all_train_file_path = f"{training_data_folder}/{start_date_str}_{end_date_str}_all.csv"
  
  all_data = []
  
  while current_date <= end_date:
    dates_between.append(current_date)
    date_str = current_date.strftime("%Y%m%d")
    print(f"training on {date_str}")
    X, y = prepare_training_data(date_str, training_data_folder)
    # merge all X, y into one file first
    # Append X and y to the respective lists
    X[target_col] = y
    # use log10 to reduce the value range
    X[target_col] = np.log10(X[target_col]+1e-2)
    
    all_data.append(X)
    current_date += timedelta(days=1)
    
  # Concatenate all X and y DataFrames vertically
  all_data_combined = pd.concat(all_data, axis=0)
  
  all_data_combined = all_data_combined.dropna(subset=['FRP'])
    
  all_data_combined.to_csv(all_train_file_path, index=False)
  
  print(f"all training data is saved to {all_train_file_path}")
  
  X = all_data_combined.drop(columns=[target_col])
  y = all_data_combined[target_col]
  
  print("input columns: ", X.columns)
  y_df = pd.DataFrame(y, columns=[target_col])
  y_df.dropna(inplace=True)
  print("get some statistics: ", y_df[target_col].describe(include='all'))
  # Define sample weights based on the threshold
  #sample_weights = np.where(y > threshold, 10.0, 1.0)  # Assign a weight of 2 to values > 100
  #sample_weights = 10 * y
  # Split the data into training and testing sets
  #X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42, shuffle=False)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

  #model.fit(X_train, y_train, sample_weight=sw_train)
  model.fit(X_train, y_train)
  
  print("y_train.shape = ", y_train.shape)
  print("y_test.shape = ", y_test.shape)
  
  # Evaluate the model on test set
  y_pred_test = model.predict(X_test)
  
  print("y_pred_test.shape = ", y_pred_test.shape)
  y_predicted_df = pd.DataFrame(y_pred_test, columns=["predicted_FRP"])
  print("get some statistics of the predicted FRP: ", y_predicted_df["predicted_FRP"].describe())
  
  # Calculate Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred_test)
  print("Mean Squared Error (MSE):", mse)
  # Calculate Root Mean Squared Error (RMSE)
  rmse = np.sqrt(mse)
  print("Root Mean Squared Error (RMSE):", rmse)

  # Calculate Mean Absolute Error (MAE)
  mae = mean_absolute_error(y_test, y_pred_test)
  print("Mean Absolute Error (MAE):", mae)

  # Calculate R-squared
  r2 = r2_score(y_test, y_pred_test)
  print("R-squared (R2):", r2)

#   label += 1

#     if label % 30 == 0:
#       # save the model for every month
#       with open(model_path, 'wb') as model_file:
#         pickle.dump(model, model_file)
  
  # Save the model to a file
  with open(model_path, 'wb') as model_file:
      pickle.dump(model, model_file)
      print(f"The new model is saved to {model_path}")
      
  now = datetime.now()
  date_time = now.strftime("%Y%d%m%H%M%S")
  random_model_path = f"{model_path}_{start_date_str}_{end_date_str}_{date_time}.pkl"
  # Save the model to a file
  with open(random_model_path, 'wb') as model_file:
      pickle.dump(model, model_file)
      print(f"The new model is saved to {random_model_path}")

  # copy a version to the latest file placeholder
  #shutil.copy(random_model_path, model_path)
  print(f"a copy of the model is saved to {model_path}")

if __name__ == "__main__":
  # Define your start and end dates as strings
  # Define your start and end dates as strings
  start_date_str = "20200109"
  end_date_str = "20201231"
  training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/all_cells_new_with_yunyao_window_data/"
  # all_cells_new_6 - this training will not use weight. Directly train the model on all rows because we already filterred out the non-fire cells.
  train_model(start_date_str, end_date_str, training_data_folder)
  print("all training on {training_data_folder} is done")
