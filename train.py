#!/usr/bin/env python3
"""Training script for running model training using PyTorch Lightning.
Supports XRDTransformer, UNet, and GAN model pipelines.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from datasets.train import get_train_dl_ds
from pl_models import TrainPipeline, XRDTransformerPipeline, GANPipeline


def load_config(config_path: Path) -> dict:
    """Load and parse YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@rank_zero_only
def check_dir(dirname: str) -> None:
    """Check if directory exists and handle accordingly.
    
    Args:
        dirname: Path to directory to check
    
    Raises:
        ValueError: If directory exists and user chooses not to overwrite
    """
    if not os.path.exists(dirname):
        return

    print(f"Save directory '{dirname}' already exists")
    print("Overwrite? [y/n]")
    ans = input().lower()
    if ans == 'y':
        shutil.rmtree(dirname)
        return

    raise ValueError("Cannot log experiment into existing directory")


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train neural networks using PyTorch Lightning.'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to YAML config file',
        default='configs/train.yaml'
    )
    return parser.parse_args(args)


def train(args=None):
    """
    Main training function.
    
    Args:
        args: Either command line arguments or config dict
    """
    # Set random seed for reproducibility
    seed_everything(42)

    # Handle arguments and load config
    if args is None:
        args = parse_args(sys.argv[1:])
        config = load_config(args.config)
    else:
        config = args

    # Set up save path
    config['save_path'] = os.path.join(
        config['exp_path'],
        config['project'],
        config['exp_name']
    )

    # Create save directory
    check_dir(config['save_path'])
    os.makedirs(config['save_path'], exist_ok=True)

    # Initialize logger
    tensorboard_logger = TensorBoardLogger(
        config['save_path'],
        name='metrics'
    )

    # Initialize data loaders
    train_loader, _ = get_train_dl_ds(config, mode='train')
    val_loader, _ = get_train_dl_ds(config, mode="val")

    # Initialize appropriate model
    if config['pipeline'] == 'xrd_transformer':
        model = XRDTransformerPipeline(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
    elif config['pipeline'] == 'unet':
        model = TrainPipeline(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )
    elif config['pipeline'] == 'gan':
        model = GANPipeline(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['save_path'],
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        **config['checkpoint']
    )

    # Initialize callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        checkpoint_callback
    ]

    # Initialize and run trainer
    trainer = Trainer(
        callbacks=callbacks,
        logger=tensorboard_logger,
        **config['trainer']
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()
