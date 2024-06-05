import os
import pandas as pd
from datetime import datetime, timedelta

# this file contains all the function that are required by both the training data preparation and testing data preparation.

# Folder path containing the text files
folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata'  # The folder yunyao provided with two years of txt files
my_file_path = "/groups/ESS3/zsun/firecasting/data/others/"
grid_to_window_mapper_csv = f"{my_file_path}/grid_cell_fixed_order_mapper.csv"
training_data_folder = "/groups/ESS3/zsun/firecasting/data/train/"


def create_grid_to_window_mapper(the_folder_path = folder_path):
  if os.path.exists(grid_to_window_mapper_csv):
    print(f"The file '{grid_to_window_mapper_csv}' exists.")
  else:
    # this function will find the nearest 24 pixels for one pixel
    # we only start from 2
    # choose any txt 
    # Replace 'path_to_folder' with the path to your folder containing text files
    txt_folder_path = f'{the_folder_path}/*.txt'
    import glob
    # Get a list of text files in the folder
    text_files = glob.glob(txt_folder_path)

    # Choose the first text file (you can modify this to select any specific file)
    file_to_read = text_files[0]

    # Read the chosen text file into a DataFrame
    df = pd.read_csv(file_to_read)  # Modify delimiter as needed
    print(df.head())
    # Convert all values in the DataFrame to numeric
    df = df.applymap(pd.to_numeric, errors='coerce')

    print(df.columns)

    # Use groupby to get unique pairs of LAT and LON
    # Use groupby to get unique pairs of LAT and LON
    unique_pairs = df.groupby(['LAT', ' LON']).size().reset_index(name='Count')
    unique_pairs_df = unique_pairs[['LAT', ' LON']]
    print("unique_pairs = ", unique_pairs_df)
    # find the nearest 24 pixels for every single pixels
    # Create a KDTree using 'LAT' and 'LON' columns
    from scipy.spatial import cKDTree
    tree = cKDTree(unique_pairs_df[['LAT', ' LON']])

    # Find the 24 nearest neighbors for each point
    distances, indices = tree.query(unique_pairs_df, k=25)

    print("distances = ", distances)

    # Extract the nearest 24 neighbors (excluding the point itself)
    nearest_24 = indices[:, 1:]
    print("nearest_24 = ", nearest_24)
    print("nearest_24.shape = ", nearest_24.shape)
    
    clockwise_indices = []
    for neighbor_indices in nearest_24:
      # Calculate angle of each neighbor with respect to current cell
      angles = []
      for neighbor_index in neighbor_indices[1:]:
        neighbor_coords = unique_pairs_df.iloc[neighbor_index]
        neighbor_coords[' LAT']
        cell_coords[' LON']

        # Sort neighbors based on angles in clockwise order
        sorted_indices = [x for _, x in sorted(zip(angles, cell_indices[1:]))]

        # Append the sorted indices to the clockwise_indices list
        clockwise_indices.append(np.concatenate(([cell_indices[0]], sorted_indices)))


    # Create column names for the new columns
    new_columns = [f'Nearest_{i}' for i in range(1, 25)]

    nearest_24_df = pd.DataFrame(nearest_24, columns=new_columns)

    print("unique_pairs_df.shape: ", unique_pairs_df.shape)

    # Merge the DataFrames row by row
    result = pd.concat([unique_pairs_df.reset_index(drop=True), nearest_24_df.reset_index(drop=True)], axis=1)

    print(result.head())
    print(result.shape)

    result.to_csv(grid_to_window_mapper_csv, index=False)
    print(f"grid to window mapper csv is saved to {grid_to_window_mapper_csv}")
    
# create_grid_to_window_mapper()
    
