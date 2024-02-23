# create a ML model for wildfire emission forecasting
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_log_error, median_absolute_error, max_error
import warnings
from datetime import datetime, timedelta
from fc_model_creation import prepare_training_data


import pickle

from fc_train_data_preprocess import read_original_txt_files, get_one_day_time_series_training_data

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

#model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v2_one_month_202007.pkl"
#model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_one_year_2020.pkl"
model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_5_days_2020_maxdepth_8_linear_weights_100_slurm_test.pkl"
output_folder_name = "output_weighted_window_xgboost_2020_year_model"
#output_folder_name = "output_xgboost_202007"


def predict(start_date_str, end_date_str):
  start_date = datetime.strptime(start_date_str, "%Y%m%d")
  end_date = datetime.strptime(end_date_str, "%Y%m%d")

  # Initialize a list to store the dates
  dates_between = []
  
  # Load the saved model
  with open(model_path, 'rb') as model_file:
      loaded_model = pickle.load(model_file)

  # create output folder
  output_folder_full_path = f'/groups/ESS3/zsun/firecasting/data/output/{output_folder_name}/'
  if not os.path.exists(output_folder_full_path):
    os.makedirs(output_folder_full_path)
      
  # Iterate through the days between start and end dates
  current_date = start_date
  label = 0
  while current_date <= end_date:
    dates_between.append(current_date)
    current_date += timedelta(days=1)
    date_str = current_date.strftime("%Y%m%d")
    print(f"generating prediction for {date_str}")
    
    X, y = prepare_training_data(date_str)
    # Make predictions
    y_pred = loaded_model.predict(X)
    y_pred[y_pred < 0] = 0

    #print("y_pred : ", y_pred)

    # merge the input and output into one df
    #print("X_test shape: ", X.shape)
    y_pred_df = pd.DataFrame(y_pred, columns=["Predicted_FRP"])

    #print("y_pred_df shape: ", y_pred_df.shape)
    #print("y_pred_df head: ", y_pred_df.head())

    #merged_df = X.join(y_pred_df)
    #merged_df = pd.concat([X, y_pred_df], axis=1)
    merged_df = X
    merged_df["Predicted_FRP"] = y_pred

    #print("merged_df shape: ", merged_df.shape)
    #print("the final merged df is: ", merged_df.head())

    # save the df to a csv for plotting
    merged_df.to_csv(f'/groups/ESS3/zsun/firecasting/data/output/{output_folder_name}/'
              f'firedata_{date_str}_predicted.txt',
              index=False)

    # Calculate metrics
    y_test = y
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    # Print the metrics
    lines_to_write = [f"Mean Absolute Error (MAE): {mae}",
                      f"Mean Squared Error (MSE): {mse}",
                      f"Root Mean Squared Error (RMSE): {rmse}",
                      f"R-squared (R2) Score: {r2}",
                      f"Explained Variance Score: {explained_var}",
                      f"Mean Squared Log Error (MSLE): {msle}",
                      f"Median Absolute Error (MedAE): {medae}",
                      f"Max Error: {max_err}",
                      f"Mean: {y.mean()}",
                      f"Median: {y.median()}",
                      f"Standard Deviation: {y.std()}",
                      f"Minimum: {y.min()}",
                      f"Maximum: {y.max()}",
                      f"Count: {y.count()}"]
    
    print(lines_to_write)
    # Specify the file path where you want to save the lines
    metric_file_path = f'/groups/ESS3/zsun/firecasting/data/output/{output_folder_name}/metrics_{date_str}_predicted.txt'

    # Open the file in write mode and write the lines
    with open(metric_file_path, 'w') as file:
        for line in lines_to_write:
            file.write(line + '\n')  # Add a newline character at the end of each line
    print(f"Metrics saved to {metric_file_path}")


start_date = "20210714"
end_date = "20210731"

predict(start_date, end_date)

