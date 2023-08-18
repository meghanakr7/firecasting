# Step 1: read and prepare the txt files by yunyao

import os
import pandas as pd

# Folder path containing the text files
folder_path = '/groups/ESS3/yli74/data/AI_Emis/firedata'  # The folder yunyao provided with two years of txt files

def read_original_txt_files():
  # Specify chunk size
  #chunk_size = 1000
  row_limit = 1000

  # Initialize an empty DataFrame
  df_list = []
  total_rows = 0

  # Traverse through files in the folder
  for filename in os.listdir(folder_path):
      if filename.endswith('.txt'):
          file_path = os.path.join(folder_path, filename)
          file_df = pd.read_csv(file_path)  # Adjust separator if needed
          #for chunk in chunk_generator:
          df_list.append(file_df)
          total_rows += len(file_df)

          if total_rows >= row_limit:
              break  # Stop reading files if row limit is reached

  # Concatenate all chunks into a single DataFrame
  final_df = pd.concat(df_list, ignore_index=True)

  # Display the DataFrame
  print(final_df)
  return final_df

# target column is current day's FRP, previous days' FRP and all the other columns are inputs

#read_original_txt_files()


