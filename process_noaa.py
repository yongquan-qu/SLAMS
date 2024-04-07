import os
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import xarray as xr

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)

url = 'https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.2xdaily.1979-2022.nc'
years = np.arange(1979, 2023)
data_var = 'olr'

# Download and process file
all_data = list()
save_path = Path(f'data/noaa/{data_var}')
save_path.mkdir(parents=True, exist_ok=True)
save_path = f'{save_path}/{data_var}.nc'
    
download_file(url, save_path)
data = xr.open_dataset(save_path)
data = data.coarsen(time=2).mean()
data = data.interp(lat=np.linspace(data.lat.max(), data.lat.min(), 120), 
                   lon=np.linspace(data.lon.min(), data.lon.max(), 240), 
                   method='cubic')
    
for timestep in tqdm(data.time):
    subdata = data.sel(time=timestep)
    yymmdd = pd.to_datetime(timestep.values).strftime('%Y%m%d')
    output_path = f'data/noaa/{data_var}/noaa_{data_var}_1.5deg_{yymmdd}.zarr'
    all_data.extend(subdata[data_var].values)

    subdata = subdata.fillna(0)
    subdata.to_zarr(output_path)

os.remove(save_path)
    
# Compute climatology
all_data = np.array(all_data)
data_mean = np.nanmean(all_data)
data_sigma = np.nanstd(all_data)

ds = xr.Dataset(
    data_vars={
        'mean': (('param'), [data_mean]),
        'sigma': (('param'), [data_sigma])
    }, 
    
    coords={'param': ('param', [data_var])}
)

ds.to_zarr(f'data/noaa/climatology/climatology_{data_var}.zarr')