import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import *
from pathlib import Path
import re
import xarray as xr
import numpy as np

from lsda import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data,
        window: int = None,
        flatten: bool = False,
    ):
        super().__init__()
        self.data = data
        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = self.data[i]

        if self.window is not None:
            i = torch.randint(0, len(x) - self.window + 1, size=())
            x = torch.narrow(x, dim=0, start=i, length=self.window)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}
        
class ERA5Dataset(Dataset):
    def __init__(
        self, 
        years: List[int],
        size: Tuple[int],
        window: int = 1,
        flatten: bool = False
    ) -> None:
        
        self.size = size
        self.window = window
        self.flatten = flatten
        self.data_dir = Path(config.DATA_DIR) / 's2s' / 'era5'
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        
        # Subset files that match with patterns (eg. years specified)
        file_paths = list()
        
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            curr_files = list(self.data_dir.glob(f'*{year}*.zarr'))
            file_paths.extend(
                [f for f in curr_files if re.match(pattern, str(f.name))]
            )
            
        self.file_paths = file_paths
        self.file_paths.sort()
        
        # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization = self.normalization.sel(param='t')
        self.normalization_mean = self.normalization['mean'].values[np.newaxis, :, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[np.newaxis, :, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = len(self.file_paths) - self.window
        return data_length

    def __getitem__(self, i):
        step_indices = [target_idx for target_idx in range(i, i + self.window)]
        
        x = list()
        for step_idx in step_indices:
            x.append(xr.open_dataset(self.file_paths[step_idx], engine='zarr'))

        x = xr.concat(x, dim='step')
        x = x[['t']].to_array().values
        x = (x - self.normalization_mean[:, np.newaxis, :, :, :]) / self.normalization_sigma[:, np.newaxis, :, :, :]
        x = torch.tensor(x)
        x = x.permute((1, 0, 2, 3, 4)) # to shape (step, param, level, lat, lon)
        x = x.flatten(1, 2) # to shape (step, param * level, lat, lon)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}
    
        
def get_latent(latent, x):
    all_z = list()
    n_samples = x.shape[0]
    
    for n in range(n_samples):
        z = latent.encoder(x[n].to(device))
        z = z.detach().cpu()
        all_z.append(z)
        
    return torch.stack(all_z)