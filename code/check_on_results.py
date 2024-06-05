# Write first python in Geoweaver
import pandas as pd

output_folder = "/groups/ESS3/zsun/firecasting/data/output/output_xgboost_window_7_days_forecasting//20210714/"

#file_path=f"{output_folder}/firedata_20210720_predicted.txt"

# file_path = "/groups/ESS3/zsun/firecasting/data/train/all_cells_new//20200715_time_series_with_window.csv"

#file_path = "/groups/ESS3/zsun/firecasting/data/train/all_cells_new_3/20200702_time_series_with_window.csv"
#file_path = "/groups/ESS3/zsun/firecasting/data/train/all_cells_new_5/20201029_time_series_with_window.csv"
file_path = "/groups/ESS3/zsun/firecasting/data/output/output_xgboost_window_7_days_forecasting//20210714/firedata_20210714_predicted.txt"

#file_path = "/groups/ESS3/yli74/data/AI_Emis/firedata/firedata_20200715.txt"

df = pd.read_csv(file_path)

print(df.head())

# Assuming you want to calculate statistics of a column named 'column_name'
#column_name = ' FRP'

# Basic statistics
#stats = df[column_name].describe()

#print(stats)
print("Summary Statistics of the DataFrame:")
#print(df.describe(include='all'))
for column in df.columns:
    print(f"\nColumn: {column}")
    if pd.api.types.is_numeric_dtype(df[column]):
        print(df[column].describe())
    else:
        print(df[column].describe(include=['object']))


