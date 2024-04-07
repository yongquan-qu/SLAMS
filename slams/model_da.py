import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml

from slams.mcs import *
from slams.nn import *
from slams.score import *
from slams.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSDA(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(LSDA, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        self.aux_features = self.model_args['all_input_size'] - self.model_args['input_size']
        self.is_flatten = True
        
        # Initialize model
        if 'convae' in self.model_args['model_name']:
            self.model = ConvEncoderDecoder(
                self.model_args['window'] * self.model_args['input_size'], 
                self.model_args['latent_channels'], 
                self.model_args['kernel_sizes'], 
                self.aux_features
            )
            
            
        elif 'sda' in self.model_args['model_name']:
            
            ## Additional adjustments for LSDA
            if 'lsda' in self.model_args['model_name']:
                self.latent = self.load_latent()
                self.channel_size = self.latent.model_args['latent_channels'][-1]
                self.is_flatten = False
                
            else:
                self.latent = None
                self.channel_size = self.model_args['input_size']
    
            self.score = MCScoreNet(self.model_args['input_size'], order=self.model_args['window'] // 2)
            self.score.kernel = LocalScoreUNet(
                    channels=self.model_args['window'] * self.channel_size,
                    with_forcing=False,
                    embedding=self.model_args['embedding'],
                    hidden_channels=self.model_args['hidden_channels'],
                    hidden_blocks=self.model_args['hidden_blocks'],
                    kernel_size=self.model_args['kernel_size'],
                    activation=torch.nn.SiLU,
                    spatial=2,
                    padding_mode='circular',
                )

            self.model = VPSDE(self.score.kernel, shape=(self.model_args['window'] * self.model_args['input_size'], 
                                                  self.data_args['size'][0], self.data_args['size'][1]))
        
        else:
            raise NotImplementedError('The specified models have yet to be implemented...')
        
        self.loss = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, kwargs = batch
        
        if 'convae' in self.model_args['model_name']:
            z = self.model.encoder(x)
            x_ = self.model.decoder(z)
            loss = self.loss(x_, x[:, :self.model_args['input_size']])
            
        elif 'lsda' in self.model_args['model_name']:
            x = get_latent(self.latent.model, x)
            x = x.flatten(1,2).to(device)
            loss = self.model.loss(x, **kwargs)
            
        elif 'sda' in self.model_args['model_name']:
            loss = self.model.loss(x, **kwargs)
            
        else:
            raise NotImplementedError('The specified models have yet to be implemented...')
    
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, kwargs = batch
        
        if 'convae' in self.model_args['model_name']:
            z = self.model.encoder(x)
            x_ = self.model.decoder(z)
            loss = self.loss(x_, x[:, :self.model_args['input_size']])
            
        elif 'lsda' in self.model_args['model_name']:
            x = get_latent(self.latent.model, x)
            x = x.flatten(1,2).to(device)
            loss = self.model.loss(x, **kwargs)
            
        elif 'sda' in self.model_args['model_name']:
            loss = self.model.loss(x, **kwargs)
            
        else:
            raise NotImplementedError('The specified models have yet to be implemented...')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.model_args['learning_rate'], 
            weight_decay=self.model_args['weight_decay']
        )
        
        lr = lambda t: 1 - (t / self.model_args['epochs'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        train_years = self.data_args['train_years']
        val_years = self.data_args['val_years']
        size = self.data_args['size']
        window = self.model_args['window']
        
        self.train_dataset = MultimodalDataset([
            ERA5Dataset(years=train_years, size=size, window=window, flatten=self.is_flatten),
            AuxDataset(years=train_years, size=size, window=window, flatten=self.is_flatten, data_path='data/cpc', data_var='precip'),
            AuxDataset(years=train_years, size=size, window=window, flatten=self.is_flatten, data_path='data/noaa', data_var='olr')
            
        ][:self.aux_features + 1])
        
        self.val_dataset = MultimodalDataset([
            ERA5Dataset(years=val_years, size=size, window=window, flatten=self.is_flatten),
            AuxDataset(years=val_years, size=size, window=window, flatten=self.is_flatten, data_path='data/cpc', data_var='precip'),
            AuxDataset(years=val_years, size=size, window=window, flatten=self.is_flatten, data_path='data/noaa', data_var='olr')
            
        ][:self.aux_features + 1])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.data_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.data_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], persistent_workers=True)
    
    def load_latent(self):
        """
        Return pre-trained autoencoder module given its name and version number
        """
        
        autoencoder_name = self.model_args['autoencoder_name']
        version_num = self.model_args['version_num']
        
        log_dir = Path('../logs') / autoencoder_name
        config_dir = Path('../slams') / 'configs'

        # Retrieve hyperparameters
        with open(config_dir / f'{autoencoder_name}.yaml', 'r') as config_filepath:
            hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)

        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']

        # Initialize model
        latent = LSDA(model_args=model_args, data_args=data_args).to(device)

        # Load model from checkpoint
        ckpt_filepath = log_dir / f'lightning_logs/version_{version_num}/checkpoints/'
        ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
        latent = latent.load_from_checkpoint(ckpt_filepath)
        
        # Ensure not-trainable
        for param in latent.parameters():
            param.requires_grad = False

        return latent
