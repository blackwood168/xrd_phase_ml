#!/usr/bin/env python3
"""
Test script for running model evaluation using PyTorch Lightning.
Supports XRDTransformer and other model pipelines.
"""

import argparse
import sys
from pathlib import Path

import yaml
from lightning.pytorch import Trainer

from datasets.test import get_test_dl_ds
from pl_models import TestPipeline, XRDTransformerTestPipeline


def load_config(config_path: Path) -> dict:
    """Load and parse YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run model evaluation using PyTorch Lightning.'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to YAML config file',
        default='configs/test.yaml'
    )
    return parser.parse_args(args)


def test(args=None):
    """
    Main testing function.
    
    Args:
        args: Either command line arguments or config dict
    """
    # Handle arguments and load config
    if args is None:
        args = parse_args(sys.argv[1:])
        config = load_config(args.config)
    else:
        config = args

    # Initialize data
    dataloader, _ = get_test_dl_ds(config)

    # Initialize appropriate model
    if config['pipeline'] == 'xrd_transformer':
        model = XRDTransformerTestPipeline(config)
    else:
        model = TestPipeline(config)

    # Set up trainer and run testing
    trainer = Trainer(logger=False, **config['trainer'])
    trainer.test(model, dataloaders=[dataloader])


if __name__ == "__main__":
    test()
