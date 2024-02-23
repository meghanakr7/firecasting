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
import warnings
import shutil
from datetime import datetime, timedelta
from fc_train_data_preprocess import training_data_folder

import pickle

from fc_train_data_preprocess import  prepare_training_data

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# final model path
model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_latest.pkl"




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
  model = XGBRegressor(n_estimators=100,
                       max_depth=8, 
                       #n_jobs = 16,
                       learning_rate=0.1,
                       eval_metric="rmse",
                       eta=0.1,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       min_child_weight=1,
                       objective='reg:squarederror')
                       #scale_pos_weight=abs((len(y_train) - y_train.sum()) / y_train.sum()))

  # Iterate through the days between start and end dates
  current_date = start_date
  label = 0
  while current_date <= end_date:
    dates_between.append(current_date)
    current_date += timedelta(days=1)
    date_str = current_date.strftime("%Y%m%d")
    print(f"training on {date_str}")
    X, y = prepare_training_data(date_str, training_data_folder)
    print("input columns: ", X.columns)
    y_df = pd.DataFrame(y, columns=['training_frp'])
    print(y_df)
    y_df.dropna(inplace=True)
    print("get some statistics: ", y_df["training_frp"].describe())
    
    # Define sample weights based on the threshold
    #sample_weights = np.where(y > threshold, 10.0, 1.0)  # Assign a weight of 2 to values > 100
    sample_weights = 10 * y
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42, shuffle=False)
    
    if label == 0:
      model.fit(X_train, y_train, sample_weight=sw_train)
      
    else:
      model.fit(X_train, y_train, xgb_model=model, sample_weight=sw_train)  # this is right incremental learning ? use the model as input to make sure the next training cycle takes all the things learnt in the previous cycles.
    
    label += 1
    
    if label % 30 == 0:
      # save the model for every month
      with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
  
  # Save the model to a file
  with open(model_path, 'wb') as model_file:
      pickle.dump(model, model_file)
      print(f"The new model is saved to {model_path}")
      
  now = datetime.now()
  date_time = now.strftime("%Y%d%m%H%M%S")
  random_model_path = f"/groups/ESS3/zsun/firecasting//model/fc_xgb_model_v1_{start_date_str}_{end_date_str}_{date_time}.pkl"
  # Save the model to a file
  with open(random_model_path, 'wb') as model_file:
      pickle.dump(model, model_file)
      print(f"The new model is saved to {random_model_path}")

  # copy a version to the latest file placeholder
  shutil.copy(random_model_path, model_path)
  print(f"a copy of the model is saved to {model_path}")

if __name__ == "__main__":
  # Define your start and end dates as strings
  start_date_str = "20200701"
  end_date_str = "20200730"
  train_model(start_date_str, end_date_str)
  print("this is just testing if it works")
