import os
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    """Dataset class for training and validation data."""

    def __init__(self, config: Dict[str, Any], mode: str = 'train') -> None:
        """Initialize the dataset.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
            mode: Either 'train' or 'val' to specify dataset mode
        """
        super().__init__()
        self.config = config
        self.mode = mode
        self.sym_groups = []
        self.annotations = []

        # Set up data directories and paths
        self.img_dir = os.path.join(
            config["data_path"],
            config['images'] if mode == 'train' else config['val_images']
        )
        csv_path = os.path.join(
            config["annotation_path"], 
            config['annotations'] if mode == 'train' else config['val_annotations']
        )

        # Initialize dataset indices
        first_index = config['first_index'] if config['first_index'] else 0
        max_index = config['max_index'] if config['max_index'] else     0
        
        # Load annotations
        self._read_csv(csv_path, first_index, max_index)
        print(f"{mode} dataset size: {len(self.annotations)}")

        # Set up lattice parameter mappings
        laue_types = {
            'romb': {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]},
            'clin': {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]},
            'all': {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}
        }
        self.hkl_minmax = laue_types[self.config['laue']]
        
        # Create index mapping dictionaries
        dics = {'h': {}, 'k': {}, 'l': {}}
        for letter in 'hkl':
            min_val, max_val = self.hkl_minmax[letter]
            for i in range(max_val - min_val + 1):
                dics[letter][min_val + i] = i
        self.h2ind, self.k2ind, self.l2ind = dics['h'], dics['k'], dics['l']

    def preprocess(self, image: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the raw image data.
        
        Args:
            image: Dictionary containing intensity and index data
            
        Returns:
            Tuple of preprocessed low and high resolution tensors
        """
        intensity = image['Intensity']
        ind_high = image['Ind_high']
        ind_low = image['Ind_low']

        # Initialize arrays
        size = [1] + [
            self.hkl_minmax[letter][1] - self.hkl_minmax[letter][0] + 1 
            for letter in 'hkl'
        ]
        low = np.zeros(size)
        high = np.zeros(size)

        # Fill arrays with intensity values
        for j, ind in enumerate(ind_high):
            h, k, l = ind
            if h in self.h2ind.keys() and k in self.k2ind.keys() and l in self.l2ind.keys():
                high[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = intensity[j]
            
        for ind in ind_low:
            h, k, l = ind
            if h in self.h2ind.keys() and k in self.k2ind.keys() and l in self.l2ind.keys():
                low[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = high[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]]

        # Normalize
        low = np.sqrt(low)
        high = np.sqrt(high)
        factor = low.max()
        high /= factor
        low /= factor

        return torch.from_numpy(low).float(), torch.from_numpy(high).float()

    def _read_csv(self, csv_path: str, first_index: int = None, max_index: int = None) -> None:
        """Read and process the CSV file containing annotations.
        
        Args:
            csv_path: Path to CSV file
            first_index: Starting index for dataset subset
            max_index: Ending index for dataset subset
        """
        df = pd.read_csv(csv_path)
        print(f"Dataset indices - First: {first_index}, Max: {max_index}")
        
        if max_index:
            df = df.iloc[first_index:max_index]
            
        for _, row in df.iterrows():
            self.annotations.append(row['filename'])

    def load_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Load a single sample from the dataset.
        
        Args:
            idx: Index of sample to load
            
        Returns:
            Dictionary containing sample data
            
        Raises:
            ValueError: If image file doesn't exist
        """
        image_path = self.annotations[idx]
        full_path = os.path.join(self.img_dir, image_path)
        
        if not os.path.exists(full_path):
            raise ValueError(f"Image not found at path: {full_path}")
            
        return np.load(full_path, allow_pickle=True)

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of item to get
            
        Returns:
            Dictionary containing preprocessed image tensors and index
        """
        img = self.load_sample(idx)
        image, image_recon = self.preprocess(image=img)
        idx = torch.as_tensor(idx).long()
        
        return {
            'image': image,
            'image_recon': image_recon,
            'idx': idx
        }


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for the dataloader.
    
    Args:
        batch: List of samples to collate
        
    Returns:
        Tuple of batched tensors
    """
    image = torch.stack([b['image'] for b in batch], dim=0)
    image_recon = torch.stack([b['image_recon'] for b in batch], dim=0)
    return image, image_recon


def get_train_dl_ds(config: Dict[str, Any], mode: str = 'train') -> Tuple[DataLoader, Dataset]:
    """Create and return training/validation dataloader and dataset.
    
    Args:
        config: Configuration dictionary
        mode: Either 'train' or 'val'
        
    Returns:
        Tuple of (dataloader, dataset)
    """
    dataset = TrainDataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
