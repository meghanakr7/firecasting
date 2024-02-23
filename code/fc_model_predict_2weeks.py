
"""
Wildfire Emission Forecasting

This script creates a machine learning model for wildfire emission forecasting and uses it to predict emissions
for a period of two weeks. The script loads a pre-trained model, processes input data, makes predictions, 
and calculates various metrics for each day within the two-week period.

The predicted emissions are saved in separate folders for each date, along with corresponding metrics.

Usage:
    python fc_model_predict_2weeks.py

Dependencies:
    - Python 3.x
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - datetime

"""

import os
import pandas as pd
import numpy as np
import dask
from dask import delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_log_error, median_absolute_error, max_error
import warnings
from datetime import datetime, timedelta
import pickle
from fc_test_data_preparation import prepare_testing_data_for_2_weeks_forecasting
from fc_train_data_preprocess import columns_to_check
import sys
from fc_model_creation import model_path
         
log_file = open('/scratch/zsun/output.log', 'w')
sys.stdout = log_file
print("This will be written to the file")
sys.stdout = sys.__stdout__  # Reset stdout to its default value

forecasting_days = 7

# Suppress the specific warning
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

#model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v2_one_month_202007.pkl"
#model_path = "/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_two_months_2020_maxdepth_7_slurm.pkl"
# model_path="/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_one_month_2020_maxdepth_15_slurm.pkl"
#model_path="/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_7_days_with_window_July2020_weighted_slurm_test.pkl"
#model_path = #"/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_one_year_2020.pkl"
output_folder_name = f"output_xgboost_window_{forecasting_days}_days_forecasting"
# output_folder_name = "output_xgboost_202007"

def remove_element_by_value(input_list, value_to_remove):
    # Check if the value is in the list
    if value_to_remove in input_list:
        # Remove the first occurrence of the value
        input_list.remove(value_to_remove)
    return input_list

#@delayed
def predict_single_day_in_the_2weeks(single_day_current_date_str, date_str, specific_date_result_folder, loaded_model):
  
    print("predicting: ", single_day_current_date_str)

    X, y = prepare_testing_data_for_2_weeks_forecasting(
      single_day_current_date_str,  # the forecasting day for the current starting day
      date_str,  # starting day
      specific_date_result_folder)
    print(f"X and y are loaded into memory for {single_day_current_date_str}")
    # Make predictions
    print("the loaded model is: ", loaded_model)
    y_pred = loaded_model.predict(X)
    print(f"Prediction for {single_day_current_date_str} of start day {date_str} is finished")
    y_pred[y_pred < 0] = 0  # if FRP is lower than 5, make it 0

    # merge the input and output into one df
    y_pred_df = pd.DataFrame(y_pred, columns=["Predicted_FRP"])

    merged_df = X
    merged_df["Predicted_FRP"] = y_pred
    
    # first remove all the rows with no fire nearby or in the past
    new_columns_to_check = remove_element_by_value(columns_to_check, " FRP")
    print("new columns to check: ", new_columns_to_check)
    print("current merged_df columns: ", merged_df.columns)
    # merged_df = merged_df[merged_df[new_columns_to_check].eq(0).all(axis=1)]

    # Define a custom function to update 'Predicted_FRP' column
    def update_FRP(row):
      # if the pixel is in ocean or predicted value is negative, FRP is 0
      if row[' VPD'] < 0 or row[' HT'] < 0 or row['Predicted_FRP'] < 0:
        return 0
      
      return row['Predicted_FRP']

    # Apply the custom function to update 'FRP' column
    merged_df['Predicted_FRP'] = merged_df.apply(update_FRP, axis=1)
    
    merged_df.loc[merged_df[new_columns_to_check].eq(0).all(axis=1), "Predicted_FRP"] = 0

    predict_file = f'{specific_date_result_folder}/firedata_{single_day_current_date_str}_predicted.txt'
    # save the df to a csv for plotting
    merged_df.to_csv(predict_file, index=False)
    print(f"Prediction results are saved to {predict_file}")

    # Calculate metrics
    y_test = y
    y_pred = merged_df['Predicted_FRP']
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    explained_var = explained_variance_score(y_test, y_pred)
    #msle = mean_squared_log_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    # Print the metrics
    lines_to_write = [f"Mean Absolute Error (MAE): {mae}",
                      f"Mean Squared Error (MSE): {mse}",
                      f"Root Mean Squared Error (RMSE): {rmse}",
                      f"R-squared (R2) Score: {r2}",
                      f"Explained Variance Score: {explained_var}",
                      #f"Mean Squared Log Error (MSLE): {msle}",
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
    metric_file_path = f'{specific_date_result_folder}/metrics_{single_day_current_date_str}_predicted.txt'

    # Open the file in write mode and write the lines
    with open(metric_file_path, 'w') as file:
      for line in lines_to_write:
        file.write(line + '\n')  # Add a newline character at the end of each line
    print(f"Metrics saved to {metric_file_path}")
    
    
#@delayed
def predict_2weeks_for_one_day(date_str, current_date, output_folder_full_path, loaded_model):
    
    # create a new folder for this date
    specific_date_result_folder = f"{output_folder_full_path}/{date_str}"
    if not os.path.exists(specific_date_result_folder):
      os.makedirs(specific_date_result_folder)
      print(f"created folder for specific date: {specific_date_result_folder}")

    # Start a loop to iterate through dates
    single_day_current_date = current_date
    # Calculate the end date (one month later)
    single_day_predict_end_date = single_day_current_date + timedelta(days=forecasting_days)
    print("single_day_current_date = ", single_day_current_date)
    print("single_day_predict_end_date = ", single_day_predict_end_date)

    parallel_tasks = []
    while single_day_current_date < single_day_predict_end_date:
      single_day_current_date_str = single_day_current_date.strftime('%Y%m%d')
      predict_single_day_in_the_2weeks(single_day_current_date_str, date_str, specific_date_result_folder, loaded_model)
      #parallel_tasks.append(predict_single_day_in_the_2weeks(single_day_current_date_str, date_str, specific_date_result_folder, loaded_model))
      single_day_current_date += timedelta(days=1)
    #print("Compute the parallel tasks")
    #dask.compute(parallel_tasks)

def predict_2weeks(start_date_str, end_date_str):
    """
    Predict wildfire emissions for a two-week period.

    Args:
        start_date_str (str): The start date in the format 'YYYYMMDD'.
        end_date_str (str): The end date in the format 'YYYYMMDD'.

    Returns:
        None
    """
    print(f"the model in use is {model_path}",)
    
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Initialize a list to store the dates
    dates_between = []

    # Load the saved model
    loaded_model = None
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # create output folder
    output_folder_full_path = f'/groups/ESS3/zsun/firecasting/data/output/{output_folder_name}/'
    if not os.path.exists(output_folder_full_path):
        os.makedirs(output_folder_full_path)

    # Iterate through the days between start and end dates
    current_date = start_date
    label = 0
    parallel_tasks = []
    while current_date <= end_date:
        print("current date: ", current_date)
        dates_between.append(current_date)
        date_str = current_date.strftime("%Y%m%d")
        predict_2weeks_for_one_day(date_str, current_date, output_folder_full_path, loaded_model)
        # parallel_tasks.append(predict_2weeks_for_one_day(date_str, current_date, output_folder_full_path, loaded_model))
        
        # increase the date by 1
        current_date += timedelta(days=1)

    # Compute the parallel tasks
    # dask.compute(parallel_tasks)

if __name__ == "__main__":
  start_date = "20210714"
  end_date = "20210715"

  predict_2weeks(start_date, end_date)


