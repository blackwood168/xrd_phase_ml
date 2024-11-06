import os
import numpy as np
import lightning as L
import pandas as pd
import torch
from torch import optim
from torchmetrics import MetricCollection, MeanSquaredError
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.functional as F

#import cctbx
#from cctbx import miller
#from cctbx import crystal
#from cctbx.array_family import flex

from losses.loss import TorchLoss
from models.model import MiniUnet, UpBlock, DownBlock, DoubleConv
import models.superformer as superformer
import models.unet_fft as unet_fft

from models.gan_model import Generator, Discriminator
from models.xrd_transformer import XRDTransformer
import torch.nn as nn


class XRDTransformerPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config
        
        self.model = XRDTransformer(
            input_shape=(26, 18, 23),  # Your tensor dimensions
            embed_dim=128,
            depth=5,
            num_heads=4,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            embedding_type='onehot'  # or 'onehot'
        )
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]#.lstrip('model.')
                #print(key_new)
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('loaded successfully')
        self.criterion = TorchLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.mse_loss = MeanSquaredError()
        self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True, spatial_dims=3)
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")
        
        # Initialize metrics storage
        #self.training_step_outputs = []
        #self.validation_step_outputs = []
        
        self.save_hyperparameters(config)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.config['optimizer_params']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['trainer']['max_epochs'],
            eta_min=self.config['optimizer_params']['lr'] * 0.01
        )
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("Loss/train", loss, prog_bar=True)
        self.train_metrics.update(out, y)
        R_factor = torch.mean(torch.sum(torch.abs(torch.abs(out)-torch.abs(y)), axis = (1, 2, 3, 4))/torch.sum(torch.abs(y), axis = (1, 2, 3, 4)))
        self.log('R/train', R_factor)
        self.log('SSIM_loss/train', 1 - self.ssim(out, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        self.valid_metrics.update(out, y)
        R_factor = torch.mean(torch.sum(torch.abs(torch.abs(out)-torch.abs(y)), axis = (1, 2, 3, 4))/torch.sum(torch.abs(y), axis = (1, 2, 3, 4)))
        self.log('R/val', R_factor)
        self.log('SSIM_loss/val', 1 - self.ssim(out, y))
        
        return loss
    
    def on_training_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class GANPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader):
        super().__init__()
        self.automatic_optimization = False  # Set manual optimization
        self.config = config
        
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        # Loss functions
        self.adversarial_criterion = nn.BCELoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('loaded successfully')
        
        # Metrics
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")
        
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        lr_g = self.config['optimizer_params'].get('lr_generator', 0.0002)
        lr_d = self.config['optimizer_params'].get('lr_discriminator', 0.0002)
        
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        return [opt_g, opt_d], []
        
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        x_low, x_high = batch
        batch_size = x_low.size(0)
        
        # Create mask for edge regions
        mask = (x_low != 0).float()
        edge_mask = 1 - mask
        
        # Train Generator
        opt_g.zero_grad()
        
        # Generate completed image
        fake_complete = self.generator(x_low)
        
        # Adversarial loss
        validity = self.discriminator(fake_complete)
        g_loss_adv = self.adversarial_criterion(validity, torch.ones_like(validity))
        
        # Reconstruction loss (focused on edge regions)
        g_loss_rec = self.reconstruction_criterion(fake_complete * edge_mask, x_high * edge_mask)
        
        # Total generator loss
        g_loss = g_loss_adv + self.config.get('lambda_rec', 100.0) * g_loss_rec
        
        self.manual_backward(g_loss)
        opt_g.step()
        
        # Train Discriminator
        opt_d.zero_grad()
        
        # Real loss
        real_validity = self.discriminator(x_high)
        d_real_loss = self.adversarial_criterion(real_validity, torch.ones_like(real_validity))
        
        # Fake loss
        fake_validity = self.discriminator(fake_complete.detach())
        d_fake_loss = self.adversarial_criterion(fake_validity, torch.zeros_like(fake_validity))
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        self.manual_backward(d_loss)
        opt_d.step()
        
        # Log losses
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("g_loss_adv", g_loss_adv)
        self.log("g_loss_rec", g_loss_rec)
        self.log("d_loss", d_loss, prog_bar=True)
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class TrainPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config

        #self.model = superformer.SuperFormer()
        #self.model = MiniUnet()
        self.model = unet_fft.UNet_FFT()
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]#.lstrip('model.')
                #print(key_new)
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('loaded successfully')
        self.criterion = TorchLoss()
        self.mse_loss = MeanSquaredError()
        self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True, spatial_dims=3)
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
        #print(x.shape)
        out = self.model(x)
        mask = x==0
        out = out*mask + x
        loss = self.criterion(out, y)
        self.log("Loss/train", loss, prog_bar=True)
        self.train_metrics.update(out, y)
        R_factor = torch.mean(torch.sum(torch.abs(torch.abs(out)-torch.abs(y)), axis = (1, 2, 3, 4))/torch.sum(torch.abs(y), axis = (1, 2, 3, 4)))
        self.log('R/train', R_factor)
        self.log('SSIM_loss/train', 1 - self.ssim(out, y))
        #self.log('MSE', self.mse_loss(out, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        mask = x==0
        out = out*mask + x
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        self.valid_metrics.update(out, y)
        R_factor = torch.mean(torch.sum(torch.abs(torch.abs(out)-torch.abs(y)), axis = (1, 2, 3, 4))/torch.sum(torch.abs(y), axis = (1, 2, 3, 4)))
        self.log('R/val', R_factor)
        self.log('SSIM_loss/val', 1 - self.ssim(out, y))
        #self.log('MSE', self.mse_loss(out, y))
 
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
        self.criterion = TorchLoss()
        self.model = superformer.SuperFormer()
        self.ssim = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=True, spatial_dims=3)
        self.mse_loss = MeanSquaredError()
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
        self.ssims = []
        self.mses = []
        self.R_factors = []
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
        R_factor = torch.mean(torch.sum(torch.abs(torch.abs(recon)-torch.abs(y)), axis = (1, 2, 3, 4))/torch.sum(torch.abs(y), axis = (1, 2, 3, 4))).cpu().numpy()
        self.R_factors.append(R_factor)
        self.test_outputs_pre2.append(self.criterion(y, recon).cpu().numpy())
        self.mses.append(self.mse_loss(y, recon).cpu().numpy())
        self.ssims.append(self.ssim(y, recon).cpu().numpy())

    def on_test_epoch_end(self):
        print(f'without pre: {np.mean(self.test_outputs)}')
        print(f'with pre1: {np.mean(self.test_outputs_pre1)}')
        print(f'with pre1 + pre2: {np.mean(self.test_outputs_pre2)}')
        print(f'mse: {np.mean(self.mses)}')
        print(f'ssim: {np.mean(self.ssims)}')
        print(f'R_factor: {np.mean(self.R_factors)}')
        '''if self.trainer.is_global_zero:
            out_class = torch.cat([o['out_class'] for o in all_test_outputs], dim=0).cpu().detach().tolist()
            idx = torch.cat([o['idx'] for o in all_test_outputs], dim=0).cpu().detach().tolist()
            df = pd.DataFrame({'idx': idx, 'label': out_class}).drop_duplicates()
            file_path = os.path.join(self.config['save_path'], self.config['test_name'], "predictions.csv")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(str(file_path), index=False)'''
