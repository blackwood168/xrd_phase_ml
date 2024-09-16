import os
import numpy as np
import lightning as L
import pandas as pd
import torch
from torch import optim
from torchmetrics import MetricCollection, MeanSquaredError

import cctbx
from cctbx import miller
from cctbx import crystal
from cctbx.array_family import flex

from losses.loss import TorchLoss
from models.model import MiniUnet, UpBlock, DownBlock, DoubleConv


class TrainPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config

        self.model = MiniUnet()
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]#.lstrip('model.')
                #print(key_new)
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('loaded successfully')
        self.criterion = MeanSquaredError()
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")

        self.train_loader = train_loader
        self.val_loader = val_loader
        # In case of DDP
        # self.num_training_steps = math.ceil(len(self.train_loader) / len(config['trainer']['devices']))
        self.num_training_steps = len(self.train_loader)

        self.save_hyperparameters(config)

    def configure_optimizers(self):
        if self.config['optimizer'] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9, nesterov=True,
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.config['optimizer']}")

        scheduler_params = self.config['scheduler_params']
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params['patience'],
                min_lr=1e-9,
                factor=scheduler_params['factor'],
                mode=scheduler_params['mode'],
                verbose=scheduler_params['verbose'],
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': scheduler_params['target_metric']
            }
        elif self.config['scheduler'] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps * scheduler_params['warmup_epochs'],
                num_training_steps=int(self.num_training_steps * self.config['trainer']['max_epochs'])
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            raise ValueError(f"Unknown scheduler name: {self.config['scheduler']}")

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        self.log("Loss/train", loss, prog_bar=True)
        self.train_metrics.update(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        self.valid_metrics.update(out, y)
 
    def on_training_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()


class TestPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = MeanSquaredError()
        self.model = MiniUnet()
        state_dict = {}
        state_old = torch.load(config['weights'])['state_dict']
        for key in state_old.keys():
            key_new = key[6:]#.lstrip('model.')
            state_dict[key_new] = state_old[key]
        self.model.load_state_dict(state_dict, strict=True)
        print('Loaded successfully')
        self.test_outputs = []
        self.test_outputs_pre1 = []
        self.test_outputs_pre2 = []
        laue_types = {'romb': {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]}, 'clin': {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]}, 'all': {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}}
        hkl_minmax = laue_types[self.config['laue']]
        dics = {'h': {}, 'k': {}, 'l': {}}
        for letter in 'hkl':
            for i in range(hkl_minmax[letter][1] - hkl_minmax[letter][0] + 1):
                dics[letter][hkl_minmax[letter][0] + i] = i
        self.h2ind, self.k2ind, self.l2ind = dics['h'], dics['k'], dics['l']
    def postprocessing1(self, low, recon):
        mask = low==0
        return recon*mask + low
    def postprocessing2(self, recon, groups):
        for idx in range(recon.shape[0]):
            group = ''.join(groups[idx].split()[:-2])
            ms = miller.build_set(
                crystal_symmetry=crystal.symmetry(
                space_group_symbol=group,
                unit_cell=(25,25,25,90,90,90)),
                anomalous_flag=False, d_min = 0.8
            )
            ms_base = ms.customized_copy(
                space_group_info = ms.space_group().build_derived_point_group().info())
            ms_all = ms_base.complete_set()
            sys_abs = ms_all.lone_set(other=ms_base)
            sys_abs_list = list(sys_abs.indices())
            for ind in sys_abs_list:
                h, k, l = ind
                if h in self.h2ind.keys() and k in self.k2ind.keys() and l in self.l2ind.keys():
                    recon[idx, 0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = 0
        return recon
        
    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_step(self, batch):
        x, y, gr = batch
        out = self.model(x)
        self.test_outputs.append(self.criterion(y, out).cpu().numpy())
        recon = self.postprocessing1(x, out)
        self.test_outputs_pre1.append(self.criterion(y, recon).cpu().numpy())
        recon = self.postprocessing2(recon, gr)
        self.test_outputs_pre2.append(self.criterion(y, recon).cpu().numpy())

    def on_test_epoch_end(self):
        print(f'without pre: {np.mean(self.test_outputs)}')
        print(f'with pre1: {np.mean(self.test_outputs_pre1)}')
        print(f'with pre1 + pre2: {np.mean(self.test_outputs_pre2)}')
        '''if self.trainer.is_global_zero:
            out_class = torch.cat([o['out_class'] for o in all_test_outputs], dim=0).cpu().detach().tolist()
            idx = torch.cat([o['idx'] for o in all_test_outputs], dim=0).cpu().detach().tolist()
            df = pd.DataFrame({'idx': idx, 'label': out_class}).drop_duplicates()
            file_path = os.path.join(self.config['save_path'], self.config['test_name'], "predictions.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(str(file_path), index=False)'''
