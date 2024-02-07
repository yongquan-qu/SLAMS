import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import *
from pathlib import Path
import re
import xarray as xr
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

from lsda import config, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        self.data_dir = Path(config.ERA_DATADIR) / 's2s' / 'era5'
        self.normalization_file = Path(config.ERA_DATADIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        
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
        

class AuxDataset(Dataset):
    def __init__(
        self,
        years: List[int],
        size: Tuple[int],
        window: int = 1,
        flatten: bool = False,
        data_path: str = '', 
        data_var: str = ''
    ) -> None:
        
        self.size = size
        self.window = window
        self.flatten = flatten
        
        self.data_path = data_path
        self.data_var = data_var
        self.data_dir = Path(self.data_path) / self.data_var
        self.normalization_file = Path(self.data_path) / 'climatology' / f'climatology_{self.data_var}.zarr'
        
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
        self.normalization = self.normalization.sel(param=self.data_var)
        self.normalization_mean = self.normalization['mean'].values
        self.normalization_sigma = self.normalization['sigma'].values
        

    def __len__(self):
        data_length = len(self.file_paths) - self.window
        return data_length

    def __getitem__(self, i):
        step_indices = [target_idx for target_idx in range(i, i + self.window)]
        
        x = list()
        for step_idx in step_indices:
            x.append(xr.open_dataset(self.file_paths[step_idx], engine='zarr'))

        x = xr.concat(x, dim='time')
        x = x[[self.data_var]].to_array().values
        x = (x - self.normalization_mean) / self.normalization_sigma
        x = torch.tensor(x)
        x = x.permute((1, 0, 2, 3)) # to shape (step, param, lat, lon)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}


class MultimodalDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.is_flatten = datasets[0].flatten

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        all_x = list()
        for dataset in self.datasets:
            data = dataset[idx]
            x, kwargs = data
            all_x.append(x)
        
        dim = 0 if self.is_flatten else 1
        x = torch.cat(all_x, dim=dim).float()

        return x, kwargs
        

def get_latent(latent, x):
    all_z = list()
    n_samples = x.shape[0]
    
    with torch.no_grad():
        for n in range(n_samples):
            z = latent.encoder(x[n].to(device))
            z = z.detach().cpu()
            all_z.append(z)
        
    return torch.stack(all_z)

def load_model_from_checkpoint(model_name, version_num):
    log_dir = Path('logs') / model_name

    # Retrieve hyperparameters
    config_filepath = Path('lsda/configs') / f'{model_name}.yaml'
    with open(config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)

    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']

    # Initialize model
    baseline = model.LSDA(model_args=model_args, data_args=data_args).to(device)

    # Load model from checkpoint
    ckpt_filepath = log_dir / f'lightning_logs/version_{version_num}/checkpoints/'
    ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
    baseline = baseline.load_from_checkpoint(ckpt_filepath)
    
    return baseline

def plot_assimilation_results(true, coarse, assimilated, param_idx, save_file=None):
    """Helper function to plot assimilation results
    """
    # Plot true
    f, ax = plt.subplots(3, len(true), figsize=(16,4))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    for N in range(true.shape[0]):
        ax[0,N].imshow(true[N,param_idx], cmap=sns.cm.icefire, vmin=-2, vmax=2)
        ax[0,N].axis('off')

    # Plot coarsen
    for N in range(coarse.shape[0]):
        ax[1,N].imshow(coarse[N,param_idx], cmap=sns.cm.icefire, vmin=-2, vmax=2)
        ax[1,N].axis('off')

    # Plot assimilation
    for M in range(assimilated.shape[0]):
        x_sample = assimilated[M]
        x_sample = x_sample.detach().cpu()

        for N in range(x_sample.shape[0]):
            ax[2,N].imshow(x_sample[N,param_idx], cmap=sns.cm.icefire, vmin=-2, vmax=2)
            ax[2,N].axis('off')
            
    if save_file != None:
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        
    plt.close()
            
            
def plot_and_compute_distributions(true, assimilated, param_idx, save_file=None):
    """Plot distributions and compute their distance
    """
    assimilated = assimilated.squeeze().detach().cpu()
    
    # Plot distributions
    f, ax = plt.subplots()
    sns.histplot(true[:,param_idx].flatten(), label='truth', stat='probability', ax=ax)
    sns.histplot(assimilated[:,param_idx].flatten(), label='assimilated', stat='probability', ax=ax)
    plt.legend();
    
    if save_file != None:
        plt.savefig(save_file, dpi=200, bbox_inches='tight')
        
    plt.close()
    
    # Compute distance
    wasserstein_d = wasserstein_distance(
        true[:,param_idx].flatten(), 
        assimilated[:,param_idx].flatten()
    )
    return wasserstein_d