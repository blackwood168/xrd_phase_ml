import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datasets.train import TrainDataset


class TestDataset(TrainDataset):
    """Dataset class for testing phase that inherits from TrainDataset."""
    
    def __init__(self, config, mode='test'):
        # Configure paths for test data
        config['val_images'] = config['images']
        config['val_annotations'] = config['annotations']
        
        super().__init__(config, mode='test')
        
        self.mode = mode
        self.config = config
        self.sym_groups = []
        self.annotations = []
        
        # Set up data directories and paths
        self.img_dir = os.path.join(config["data_path"], self.config['images'])
        csv_path = os.path.join(config["annotation_path"], self.config['annotations'])

        # Handle dataset indexing
        first_index = config['first_index'] if config['first_index'] else 0
        max_index = config['max_index'] if config['max_index'] else 0

        self._read_csv(csv_path, first_index, max_index)
        
    def _read_csv(self, csv_path, first_index=None, max_index=None):
        """Read and process the CSV file containing annotations."""
        df = pd.read_csv(csv_path)
        print(f"Dataset indices - First: {first_index}, Max: {max_index}")
        
        if max_index:
            df = df.iloc[first_index:max_index]
            
        for _, row in df.iterrows():
            self.annotations.append(row['filename'])

    def load_sample(self, idx):
        """Load a single sample from the dataset."""
        image_path = self.annotations[idx]
        full_path = os.path.join(self.img_dir, image_path)
        
        if not os.path.exists(full_path):
            raise ValueError(f"Image not found at path: {full_path}")
            
        image = np.load(full_path, allow_pickle=True)
        return image

    def __getitem__(self, idx):
        """Get a single item from the dataset with preprocessing applied."""
        img = self.load_sample(idx)
        image, image_recon = self.preprocess(image=img)
        idx = torch.as_tensor(idx).long()
        
        return {
            'image': image,
            'image_recon': image_recon,
            'idx': idx
        }


def collate_fn(batch):
    """Collate function for the dataloader to stack batch items."""
    image = torch.stack([b['image'] for b in batch], dim=0)
    image_recon = torch.stack([b['image_recon'] for b in batch], dim=0)
    return image, image_recon


def get_test_dl_ds(config):
    """Create and return test dataloader and dataset."""
    dataset = TestDataset(config)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
