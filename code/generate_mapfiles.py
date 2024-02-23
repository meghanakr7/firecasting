# Generate new map files for new geotifs

import os
import shutil
import re
import pandas as pd
from datetime import datetime

#from plot_results import output_folder

def create_mapserver_map_config(target_geotiff_file_path, force=False):
  geotiff_file_name = os.path.basename(target_geotiff_file_path)
  geotiff_dir_name = os.path.dirname(target_geotiff_file_path)
  geotiff_mapserver_file_path = f"{geotiff_dir_name}/{geotiff_file_name}.map"
  if not geotiff_file_name.endswith(".tif"):
    print(f"{geotiff_file_name} is not geotiff")
    return
  
  if os.path.exists(geotiff_mapserver_file_path) and not force:
    print(f"{geotiff_mapserver_file_path} already exists")
    return geotiff_mapserver_file_path
  
  # Define a regular expression pattern to match the date in the filename
  pattern = r"\d{4}\d{2}\d{2}"

  # Use re.search to find the match
  match = re.search(pattern, geotiff_file_name)

  # Check if a match is found
  if match:
      date_string = match.group()
      print("Date:", date_string)
  else:
      print("No date found in the filename.")
      return f"The file's name {target_geotiff_file} is wrong"
  
#   Driver: GTiff/GeoTIFF
# Files: firedata_20210725_predicted.txt_output.tif
# Size is 6000, 2600
# Coordinate System is:
# GEOGCS["WGS 84",
#     DATUM["WGS_1984",
#         SPHEROID["WGS 84",6378137,298.257223563,
#             AUTHORITY["EPSG","7030"]],
#         AUTHORITY["EPSG","6326"]],
#     PRIMEM["Greenwich",0],
#     UNIT["degree",0.0174532925199433],
#     AUTHORITY["EPSG","4326"]]
# Origin = (-126.000000000000000,50.500000000000000)
# Pixel Size = (0.010000000000000,-0.010000000000000)
# Metadata:
#   AREA_OR_POINT=Area
# Image Structure Metadata:
#   INTERLEAVE=BAND
# Corner Coordinates:
# Upper Left  (-126.0000000,  50.5000000) (126d 0' 0.00"W, 50d30' 0.00"N)
# Lower Left  (-126.0000000,  24.5000000) (126d 0' 0.00"W, 24d30' 0.00"N)
# Upper Right ( -66.0000000,  50.5000000) ( 66d 0' 0.00"W, 50d30' 0.00"N)
# Lower Right ( -66.0000000,  24.5000000) ( 66d 0' 0.00"W, 24d30' 0.00"N)
# Center      ( -96.0000000,  37.5000000) ( 96d 0' 0.00"W, 37d30' 0.00"N)
# Band 1 Block=6000x1 Type=Float32, ColorInterp=Gray
  
  mapserver_config_content = f"""
MAP
  NAME "wildfiremap"
  STATUS ON
  EXTENT -126 24.5 -66 50.5
  SIZE 6000 2600
  UNITS DD
  SHAPEPATH "/var/www/html/wildfire_site/data"

  PROJECTION
    "init=epsg:4326"
  END

  WEB
    IMAGEPATH "/temp/"
    IMAGEURL "/temp/"
    METADATA
      "wms_title" "Wildfire MapServer WMS"
      "wms_onlineresource" "http://geobrain.csiss.gmu.edu/cgi-bin/mapserv?map=/var/www/html/wildfire_site/data/wildfire.map&"
      WMS_ENABLE_REQUEST      "*"
      WCS_ENABLE_REQUEST      "*"
      "wms_srs" "epsg:5070 epsg:4326 epsg:3857"
    END
  END


  LAYER
    NAME "predicted_wildfire_{date_string}"
    TYPE RASTER
    STATUS DEFAULT
    DATA "/var/www/html/wildfire_site/data/{geotiff_file_name}"

    PROJECTION
      "init=epsg:4326"
    END

    METADATA
      "wms_include_items" "all"
    END
    PROCESSING "SCALE=0.0,30.0"
    PROCESSING "SCALE_BUCKETS=15"
    PROCESSING "NODATA=0"
    STATUS ON
    DUMP TRUE
    TYPE RASTER
    OFFSITE 0 0 0
    CLASSITEM "[pixel]"
    TEMPLATE "template.html"
    INCLUDE "legend_wildfire.map"
  END
END
"""
  
  with open(geotiff_mapserver_file_path, "w") as file:
    file.write(mapserver_config_content)
    
  print(f"Mapserver config is created at {geotiff_mapserver_file_path}")
  return geotiff_mapserver_file_path

def refresh_available_date_list(target_geotiff_file_path):
  geotiff_dir_name = os.path.dirname(target_geotiff_file_path)
  
  # Define columns for the DataFrame
  columns = ["date", "predicted_wildfire_url_prefix"]

  # Create an empty DataFrame with columns
  df = pd.DataFrame(columns=columns)
  
  for filename in os.listdir(geotiff_dir_name):
    target_geotiff_file = os.path.join(geotiff_dir_name, filename)
    
    if ".tif" not in target_geotiff_file:
      continue
      
    print("Processing ", target_geotiff_file)
      
    # generate map file first
    create_mapserver_map_config(target_geotiff_file, force=True)
    
    date_str = re.search(r"\d{4}\d{2}\d{2}", filename).group()
    date = datetime.strptime(date_str, "%Y%m%d")
    formatted_date = date.strftime("%Y-%m-%d")
    # Append a new row to the DataFrame
    df = df.append({
      "date": formatted_date, 
      "predicted_wildfire_url_prefix": f"{filename}"
    }, ignore_index=True)
  
  # Save DataFrame to a CSV file
  df.to_csv(f"{geotiff_dir_name}/date_list.csv", index=False)

  # Display the final DataFrame
  print(df)
  



def generate_mapfiles():
  output_folder = "/groups/ESS3/zsun/firecasting/data/output/output_xgboost_window_15_days_forecasting/20210714/"
  print("Current folder: ", output_folder)
  refresh_available_date_list(output_folder)

if __name__ == "__main__":
  generate_mapfiles()


