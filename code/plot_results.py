# traverse the output folder and create a PNG for every day
# this doesn't use parallelization at all so it will be slow. 
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from fc_model_predict_2weeks import output_folder_name
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from scipy.interpolate import griddata

output_folder = f"/groups/ESS3/zsun/firecasting/data/output/{output_folder_name}/20210714/"
sample_lat_lon_csv = "/groups/ESS3/zsun/firecasting/data/others/sample_lat_lon.csv"


def save_predicted_frp_to_standard_netcdf(csv_file, sample_lat_lon_df):
    """
    Get the ML results ready for downstream model inputs
    """
    
    pass


def save_predicted_frp_to_geotif(csv_file, sample_lat_lon_df):
    """
    Get the ML results ready for public download and access
    """
    # Read CSV into GeoDataFrame
    df = pd.read_csv(csv_file)
    if 'LAT' not in df.columns:
      # Merge 'lat' and 'lon' columns from df2 into df1
      df["LAT"] = sample_lat_lon_df["LAT"]
      df[" LON"] = sample_lat_lon_df[" LON"]

    # Create GeoDataFrame with Point geometries
    #geometry = [Point(lon, lat) for lon, lat in zip(df[' LON'], df['LAT'])]
    #gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[" LON"], df["LAT"]), crs='EPSG:4326')
	
    # Set up rasterization parameters
    # Define pixel size and latitude/longitude bounds
    pixel_size = 0.01  # Adjust as needed
    min_lat, max_lat = gdf['LAT'].min(), gdf['LAT'].max()
    min_lon, max_lon = gdf[' LON'].min(), gdf[' LON'].max()
    
    print("min_lat = ", min_lat)
    print("max_lat = ", max_lat)
    print("min_lon = ", min_lon)
    print("max_lon = ", max_lon)

    # Calculate width and height based on pixel size
    width = int((max_lon - min_lon) / pixel_size)
    height = int((max_lat - min_lat) / pixel_size)
    
    print("width = ", width)
    print("height = ", height)
    
    # Create a regular grid using numpy
    xi = np.linspace(min_lon, max_lon, width)
    yi = np.linspace(max_lat, min_lat, height)
    xi, yi = np.meshgrid(xi, yi)
    
    print("xi = ", xi)
    print("yi = ", yi)

    # Interpolate data onto the regular grid
    zi = griddata((gdf[' LON'], gdf['LAT']), gdf['Predicted_FRP'], (xi, yi), method='linear')

    # Create a GeoTIFF from the interpolated data
    with rasterio.open(f'{csv_file}_output.tif', 
                       'w', 
                       driver='GTiff', 
                       height=height, 
                       width=width,
                       count=1, 
                       dtype='float32', 
                       crs='EPSG:4326',
                       transform=from_origin(min_lon, 
                                             max_lat, 
                                             pixel_size, 
                                             pixel_size)) as dst:
        dst.write(zi, 1)

    print(f"GeoTIFF file created: {csv_file}_output.tif")
    

def plot_png(file_path, sample_lat_lon_df):
    # Read CSV into a DataFrame
    df = pd.read_csv(file_path)
    print(df.head())

    if 'LAT' not in df.columns:
      # Merge 'lat' and 'lon' columns from df2 into df1
      df["LAT"] = sample_lat_lon_df["LAT"]
      df[" LON"] = sample_lat_lon_df[" LON"]

    real_col_num = len(df.columns) - 2
    num_rows = int(np.ceil(np.sqrt(real_col_num)))
    num_cols = int(np.ceil(real_col_num / num_rows))

    # Create a figure and axis objects
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(28, 24))
    # Flatten the axs array if it's more than 1D
    axs = np.array(axs).flatten()
    
    for i in range(len(df.columns)):
        col_name = df.columns[i]
        
        if col_name in ["LAT", " LON"]:
          continue
        
        ax = axs[i]
        # Create a scatter plot using two columns from the DataFrame
        cmap = plt.get_cmap('hot')  # You can choose a different colormap
        sm = ScalarMappable(cmap=cmap)
        if "FRP" in col_name or "Near" in col_name:
          # Define the minimum and maximum values for the color scale
          min_value = 0  # Set your minimum value here
          max_value = 50  # Set your maximum value here
          # Create a color map and a normalization for the color scale
          norm = Normalize(vmin=min_value, vmax=max_value)

          ax.scatter(
            df[' LON'], 
            df['LAT'], 
            c=df[col_name], 
            cmap=cmap, 
            s=5, 
            norm=norm,
            edgecolors='none'
          )
          # Create a scalar mappable for the color bar
          sm = ScalarMappable(cmap=cmap, norm=norm)
        else:
          min_value = df[col_name].min()
          if min_value == -999:
            min_value = 0
          
          max_value = df[col_name].max()
          #cmap = plt.get_cmap('coolwarm')  # You can choose a different colormap
          new_norm = Normalize(vmin=min_value, vmax=max_value)
          sm = ScalarMappable(cmap=cmap, norm=new_norm)
          ax.scatter(
            df[' LON'], 
            df['LAT'], 
            c=df[col_name], 
            cmap=cmap, 
            norm=new_norm,
            s=5,
            edgecolors='none'
          )
          sm.set_array([])  # Set an empty array for the color bar

        # Set the color bar's minimum and maximum values
        # Add a color bar to the plot
        color_bar = plt.colorbar(sm, orientation='horizontal', ax=ax)

        # Set the color bar's minimum and maximum values using vmin and vmax
        color_bar.set_ticks([min_value, max_value])
        color_bar.set_ticklabels([min_value, max_value])

        ax.set_title(f'{col_name}')

        # Add labels and legend
        #ax.set_xlabel('Longitude')
        #ax.set_ylabel('Latitude')

    plt.tight_layout()

    res_png_path = f"{file_path}.png"
    plt.savefig(res_png_path)
    print(f"test image is saved at {res_png_path}")
    plt.close()
    

def plot_images():
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(output_folder) if f.endswith('.txt') and f.startswith('firedata_')]
    
    sample_lat_lon_df = pd.read_csv(sample_lat_lon_csv)
    
    # Iterate through each CSV file
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(output_folder, csv_file)
        plot_png(file_path, sample_lat_lon_df)
        save_predicted_frp_to_geotif(file_path, sample_lat_lon_df)
        save_predicted_frp_to_standard_netcdf(file_path, sample_lat_lon_df)
        

    print("All done")

    
if __name__ == "__main__":
    plot_images()

