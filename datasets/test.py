import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datasets.train import TrainDataset


class TestDataset(TrainDataset):
    def __init__(self, config, mode='test'):
        config['val_images'] = config['images']
        config['val_annotations'] = config['annotations']
        super().__init__(config, mode='test')
        self.mode = mode
        self.config = config
        self.sym_groups = []
        self.annotations = []
        self.img_dir = os.path.join(config["data_path"], self.config['images'])
        csv_path = os.path.join(config["annotation_path"], self.config['annotations'])
        self._read_csv(csv_path)
        
    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            self.annotations.append(row['filename'])
            self.sym_groups.append(row['group'])
    def load_sample(self, idx):
        image_path = self.annotations[idx]
        if not os.path.exists(os.path.join(self.img_dir, image_path)):
            raise ValueError(f"{os.path.join(self.img_dir, image_path)} doesn't exist")
        #print(image)
        image = np.load(os.path.join(self.img_dir, image_path), allow_pickle = True)
        return image

    def __getitem__(self, idx):
        #print(idx)
        img = self.load_sample(idx)
        #print(len(img))
        image, image_recon = self.preprocess(image = img)
        idx = torch.as_tensor(idx).long()
        return {'image': image, 'image_recon': image_recon, 'idx': idx, 'group': self.sym_groups[idx]}


def collate_fn(batch):
    image = torch.stack([b['image'] for b in batch], dim=0)
    image_recon = torch.stack([b['image_recon'] for b in batch], dim=0)
    group = [b['group'] for b in batch]
    return image, image_recon, group


def get_test_dl_ds(config):
    dataset = TestDataset(config)

    dataloader = DataLoader(
        dataset, collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
