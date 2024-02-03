import os
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)

base_url = 'https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip'
years = np.arange(1979, 2023)
data_var = 'precip'

# Download and process file
all_data = list()
for year in tqdm(years):
    url = f'{base_url}.{year}.nc'
    save_path = f'data/cpc/{data_var}/{data_var}.{year}.nc'
    
    download_file(url, save_path)
    data = xr.open_dataset(save_path)
    data = data.coarsen(lat=3, lon=3).mean()
    
    for timestep in data.time:
        subdata = data.sel(time=timestep)
        mmdd = pd.to_datetime(timestep.values).strftime('%m%d')
        output_path = f'data/cpc/{data_var}/cpc_{data_var}_1.5deg_{year}{mmdd}.zarr'
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

ds.to_zarr(f'data/cpc/climatology/climatology_{data_var}.zarr')