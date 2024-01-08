import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import *

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
        
def get_latent(latent, x):
    all_z = list()
    n_samples = x.shape[0]
    
    for n in range(n_samples):
        z = latent.encoder(x[n].to(device))
        z = z.detach().cpu()
        all_z.append(z)
        
    return torch.stack(all_z)