import os
import argparse
from pathlib import Path
import yaml

import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
pl.seed_everything(42)

from slams import model_da


def main(args):
    """
    Training script given .yaml config for operational ERA5
    Example usage:
        (Training encoder-decoder)      1) `python train_da.py --config_filepath slams/configs/convae_0.yaml`
        (Training Pixel-DA)             2) `python train_da.py --config_filepath slams/configs/sda_0.yaml`
        (Training Latent-DA)            3) `python train_da.py --config_filepath slams/configs/lsda_0.yaml`
        
    The number at the end of the model corresponds to the number of auxiliary variables included (for multimodal)
    So SDA only has _0 suffix (ie. no observational constraints, only background states)
    """
    
    # Retrieve hyperparameters
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
        
    # Initialize model
    baseline = model_da.LSDA(model_args=model_args, data_args=data_args)
    baseline.setup()
    
    # Initialize training
    log_dir = Path('logs') / model_args['model_name']
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        strategy='auto',
        max_epochs=model_args['epochs'],
        logger=tb_logger,
        callbacks=[checkpoint_callback]
     )

    trainer.fit(baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', help='Provide the filepath string to the model config...')
    
    args = parser.parse_args()
    main(args)
