import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.sym_groups = []
        self.annotations = []
        self.mode = mode
        self.img_dir = os.path.join(
            config["data_path"],
            self.config['images'] if (mode == 'train') else self.config['val_images']
        )
        csv_path = os.path.join(
            config["annotation_path"],
            self.config['annotations'] if (mode == 'train') else self.config['val_annotations']
        )
        self._read_csv(csv_path)
        laue_types = {'romb': {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]}, 'clin': {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]}, 'all': {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}}
        self.hkl_minmax = laue_types[self.config['laue']]
        dics = {'h': {}, 'k': {}, 'l': {}}
        for letter in 'hkl':
            for i in range(self.hkl_minmax[letter][1] - self.hkl_minmax[letter][0] + 1):
                dics[letter][self.hkl_minmax[letter][0] + i] = i
        self.h2ind, self.k2ind, self.l2ind = dics['h'], dics['k'], dics['l']
    
    def preprocess(self, image):
        intensity, ind_high, ind_low = image['Intensity'], image['Ind_high'], image['Ind_low']
        # romb : {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]}
        # all : {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}
        # clin: {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]}
        size = [1]
        for letter in 'hkl':
            size.append(self.hkl_minmax[letter][1] - self.hkl_minmax[letter][0] + 1)
        low, high = np.zeros(size), np.zeros(size)
        #print(low.shape)
        #print(h2ind)
        for j, ind in enumerate(ind_high):
            h, k, l = ind
            high[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = intensity[j]
        for ind in ind_low:
            h, k, l = ind
            low[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = high[0, self.h2ind[h], self.k2ind[k], self.l2ind[l]]
        low, high = np.sqrt(low), np.sqrt(high)
        factor = low.max()
        high /= factor
        low /= factor
        return torch.from_numpy(low).float(), torch.from_numpy(high).float()
    
    
    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            self.annotations.append(row['filename'])


    def __len__(self):
        return len(self.annotations)

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
        return {'image': image, 'image_recon': image_recon, 'idx': idx}


def collate_fn(batch):
    image = torch.stack([b['image'] for b in batch], dim=0)
    image_recon = torch.stack([b['image_recon'] for b in batch], dim=0)
    return image, image_recon


def get_train_dl_ds(
        config,
        mode='train'
):
    dataset = TrainDataset(
        config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
