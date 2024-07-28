import os
import pickle
import matplotlib.pyplot as plt
from fc_model_creation import model_path

# get one prediction and reverse inference the model and get explanation for that prediction

# Explanable AI - SHAP

# target_predict_file = "/groups/ESS3/zsun/firecasting/data/output/output_xgboost_2020_two_months/20210714/firedata_20210714_predicted.txt"

#model_path="/groups/ESS3/zsun/firecasting/model/fc_xgb_model_v1_weighted_5_days_2020_maxdepth_8_linear_weights_100_slurm_test.pkl"

# # Load the saved model
# with open(model_path, 'rb') as model_file:
#   loaded_model = pickle.load(model_file)

# df = pd.read_csv(target_predict_file)
  
# X, y = prepare_testing_data_for_2_weeks_forecasting(single_day_current_date_str, date_str, specific_date_result_folder)

# # Make predictions
# y_pred = loaded_model.predict(X)

# calculate feature importance - indirect evaluation


    
def plot_feature_importance():
  
    # Load the saved model
    with open(model_path, 'rb') as model_file:
      loaded_model = pickle.load(model_file)
    
    feature_importances = loaded_model.feature_importances_
    
#     feature_names = ['FWI', 'VPD', 'HT', 'T', 'RH', 'U', 'V', 'P', 'RAIN', 'CAPE', 'ST',
#  'SM', 'Nearest_1', 'Nearest_2', 'Nearest_3', 'Nearest_4', 'Nearest_5',
#  'Nearest_6', 'Nearest_7', 'Nearest_8', 'Nearest_9', 'Nearest_10',
#  'Nearest_11', 'Nearest_12', 'Nearest_13', 'Nearest_14', 'Nearest_15',
#  'Nearest_16', 'Nearest_17', 'Nearest_18', 'Nearest_19', 'Nearest_20',
#  'Nearest_21', 'Nearest_22', 'Nearest_23', 'Nearest_24',
#  'FRP_1_days_ago', 'FRP_2_days_ago', 'FRP_3_days_ago', 'FRP_4_days_ago',
#  'FRP_5_days_ago', 'FRP_6_days_ago', 'FRP_7_days_ago']

    feature_names = ['LAT', 'LON', 'FWI', 'VPD', 'HT', 'T', 'RH', 'U', 'V', 'P', 'RAIN', 'CAPE', 'ST',
'SM', 'Nearest_1', 'Nearest_2', 'Nearest_3', 'Nearest_4', 'Nearest_5',
'Nearest_6', 'Nearest_7', 'Nearest_8', 'Nearest_9', 'Nearest_10',
'Nearest_11', 'Nearest_12', 'Nearest_13', 'Nearest_14', 'Nearest_15',
'Nearest_16', 'Nearest_17', 'Nearest_18', 'Nearest_19', 'Nearest_20',
'Nearest_21', 'Nearest_22', 'Nearest_23', 'Nearest_24', 'Land_Use', 'VCI_AVE', 'TCI_AVE', 'VHI_AVE', 'VCI_TOT', 'TCI_TOT', 'VHI_TOT',
'FWI_1_days_ago', 'VPD_1_days_ago', 'P_1_days_ago', 'FRP_1_days_ago',
'FWI_2_days_ago', 'VPD_2_days_ago', 'P_2_days_ago', 'FRP_2_days_ago',
'FWI_3_days_ago', 'VPD_3_days_ago', 'P_3_days_ago', 'FRP_3_days_ago',
'FWI_4_days_ago', 'VPD_4_days_ago', 'P_4_days_ago', 'FRP_4_days_ago',
'FWI_5_days_ago', 'VPD_5_days_ago', 'P_5_days_ago', 'FRP_5_days_ago',
'FWI_6_days_ago', 'VPD_6_days_ago', 'P_6_days_ago', 'FRP_6_days_ago',
'FWI_7_days_ago', 'VPD_7_days_ago', 'P_7_days_ago', 'FRP_7_days_ago']

# ['FWI', 'VPD', 'HT', 'T', 'RH', 'U', 'V', 'P', 'RAIN', 'CAPE', 'ST',
# > 'SM', 'Nearest_1', 'Nearest_2', 'Nearest_3', 'Nearest_4', 'Nearest_5',
# > 'Nearest_6', 'Nearest_7', 'Nearest_8', 'Nearest_9', 'Nearest_10',
# > 'Nearest_11', 'Nearest_12', 'Nearest_13', 'Nearest_14', 'Nearest_15',
# > 'Nearest_16', 'Nearest_17', 'Nearest_18', 'Nearest_19', 'Nearest_20',
# > 'Nearest_21', 'Nearest_22', 'Nearest_23', 'Nearest_24', 'Land_Use',
# > 'VCI_AVE', 'TCI_AVE', 'VHI_AVE', 'VCI_TOT', 'TCI_TOT', 'VHI_TOT',
# > 'FWI_1_days_ago', 'VPD_1_days_ago', 'P_1_days_ago', 'FRP_1_days_ago',
# > 'FWI_2_days_ago', 'VPD_2_days_ago', 'P_2_days_ago', 'FRP_2_days_ago',
# > 'FWI_3_days_ago', 'VPD_3_days_ago', 'P_3_days_ago', 'FRP_3_days_ago',
# > 'FWI_4_days_ago', 'VPD_4_days_ago', 'P_4_days_ago', 'FRP_4_days_ago',
# > 'FWI_5_days_ago', 'VPD_5_days_ago', 'P_5_days_ago', 'FRP_5_days_ago',
# > 'FWI_6_days_ago', 'VPD_6_days_ago', 'P_6_days_ago', 'FRP_6_days_ago',
# > 'FWI_7_days_ago', 'VPD_7_days_ago', 'P_7_days_ago', 'FRP_7_days_ago',
# > 'Predicted_FRP']

#     [
# #       'LAT', ' LON', 
#       'FWI', 'VPD', 'HT', 'T', 'RH', 'U', 'V', 'P',
#  'RAIN', 'CAPE', 'ST', 'SM', 'FRP_1_days_ago', 'FRP_2_days_ago',
#  'FRP_3_days_ago', 'FRP_4_days_ago', 'FRP_5_days_ago',
#  'FRP_6_days_ago', 'FRP_7_days_ago', 'Nearest_1', 'Nearest_2',
#  'Nearest_3', 'Nearest_4', 'Nearest_5', 'Nearest_6', 'Nearest_7',
#  'Nearest_8', 'Nearest_9', 'Nearest_10', 'Nearest_11', 'Nearest_12',
#  'Nearest_13', 'Nearest_14', 'Nearest_15', 'Nearest_16', 'Nearest_17',
#  'Nearest_18', 'Nearest_19', 'Nearest_20', 'Nearest_21', 'Nearest_22',
#  'Nearest_23', 'Nearest_24']
    

    # Create a bar plot of feature importances
    plt.figure(figsize=(12, 12))
    print(feature_names)
    print(feature_importances)
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance Plot')
    file_name = os.path.basename(model_path)
    new_png_path = f'/groups/ESS3/zsun/firecasting/data/output/importance_summary_plot_{file_name}.png'
    plt.savefig(new_png_path)
    print("new_png_path = ", new_png_path)
    
    
if __name__ == "__main__":
    plot_feature_importance()

# explain why it makes that decision (look into the model itself) - direct evaluation


# local explanation 


# global explanation




