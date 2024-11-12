import os
import numpy as np
import lightning as L
import pandas as pd
import torch
from torch import optim
from torchmetrics import MetricCollection, MeanSquaredError
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.functional as F
import torch.nn as nn

from losses.loss import TorchLoss
from models.model import MiniUnet, UpBlock, DownBlock, DoubleConv
import models.superformer as superformer
import models.unet_fft as unet_fft
from models.gan_model import Generator, Discriminator
from models.xrd_transformer import XRDTransformer


class XRDTransformerPipeline(L.LightningModule):
    """Pipeline for training XRD Transformer model."""
    
    def __init__(self, config, train_loader, val_loader) -> None:
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        super().__init__()
        self.config = config
        
        # Initialize model
        self.model = XRDTransformer(
            input_shape=(26, 18, 23),
            embed_dim=128,
            depth=5,
            num_heads=4,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            embedding_type='onehot'
        )
        
        # Load pre-trained weights if specified
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('Weights loaded successfully')
            
        # Initialize loss functions and metrics
        self.criterion = TorchLoss()
        self.mse_loss = MeanSquaredError()
        self.ssim = SSIM(
            data_range=1, 
            size_average=True, 
            channel=1,
            nonnegative_ssim=True, 
            spatial_dims=3
        )
        
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")
        
        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
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
        """Execute a single training step."""
        x, y = batch
        out = self.model(x)
        
        # Calculate losses
        loss = self.criterion(out, y)
        self.log("Loss/train", loss, prog_bar=True)
        
        # Update metrics
        self.train_metrics.update(out, y)
        
        # Calculate R-factor
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(out) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        )
        
        # Log metrics
        self.log('R/train', R_factor)
        self.log('SSIM_loss/train', 1 - self.ssim(out, y))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        x, y = batch
        out = self.model(x)
        
        # Calculate losses
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        
        # Update metrics
        self.valid_metrics.update(out, y)
        
        # Calculate R-factor
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(out) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        )
        
        # Log metrics
        self.log('R/val', R_factor)
        self.log('SSIM_loss/val', 1 - self.ssim(out, y))
        
        return loss
    
    def on_training_epoch_end(self):
        """Compute and log training metrics at the end of each epoch."""
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()

    def train_dataloader(self):
        """Return the training dataloader."""
        return self.train_loader

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.val_loader


class GANPipeline(L.LightningModule):
    """Pipeline for training GAN model."""

    def __init__(self, config, train_loader, val_loader):
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            train_loader: Training data loader 
            val_loader: Validation data loader
        """
        super().__init__()
        self.automatic_optimization = False  # Set manual optimization
        self.config = config
        
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        # Initialize metrics
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")
        
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        lr_g = self.config['optimizer_params'].get('lr_generator', 0.0002)
        lr_d = self.config['optimizer_params'].get('lr_discriminator', 0.0002)
        
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        return [opt_g, opt_d], []
        
    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        opt_g, opt_d = self.optimizers()
        
        x_low, x_high = batch
        batch_size = x_low.size(0)
        
        # Create mask for edge regions
        # x_low contains zeros in edge regions that need to be reconstructed
        # Mask is 1 where we have original values (center) and 0 where we need to reconstruct (edges)
        center_mask = (x_low != 0).float()
        
        # Generate completed image
        fake_complete = self.generator(x_low)
        
        # Combine generated edges with original center
        # Keep original values where center_mask is 1, use generated values where center_mask is 0
        combined_output = center_mask * x_low + (1 - center_mask) * fake_complete
        
        # Train Generator
        opt_g.zero_grad()
        
        # Adversarial loss on combined output
        validity = self.discriminator(combined_output)
        g_loss_adv = self.adversarial_criterion(validity, torch.ones_like(validity))
        
        # Reconstruction loss only on edge regions
        g_loss_rec = self.reconstruction_criterion(
            fake_complete * (1 - center_mask), 
            x_high * (1 - center_mask)
        )
        
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
        fake_validity = self.discriminator(combined_output.detach())
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
        
        # Update metrics using combined output
        self.train_metrics.update(combined_output, x_high)
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(combined_output) - torch.abs(x_high)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(x_high), axis=(1, 2, 3, 4))
        )
        self.log('R/train', R_factor)
        self.log('SSIM_loss/train', 1 - self.ssim(combined_output, x_high))
        
    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        x_low, x_high = batch
        
        # Create mask for center region
        center_mask = (x_low != 0).float()
        
        # Generate completed image
        fake_complete = self.generator(x_low)
        
        # Combine generated edges with original center
        combined_output = center_mask * x_low + (1 - center_mask) * fake_complete
        
        # Calculate validation metrics
        validity = self.discriminator(combined_output)
        g_loss_adv = self.adversarial_criterion(validity, torch.ones_like(validity))
        
        # Reconstruction loss only on edge regions
        g_loss_rec = self.reconstruction_criterion(
            fake_complete * (1 - center_mask), 
            x_high * (1 - center_mask)
        )
        
        g_loss = g_loss_adv + self.config.get('lambda_rec', 100.0) * g_loss_rec
        
        # Log validation metrics
        self.log("g_loss/val", g_loss, prog_bar=True)
        self.log("g_loss_adv/val", g_loss_adv)
        self.log("g_loss_rec/val", g_loss_rec)
        
        self.valid_metrics.update(combined_output, x_high)
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(combined_output) - torch.abs(x_high)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(x_high), axis=(1, 2, 3, 4))
        )
        self.log('R/val', R_factor)
        self.log('SSIM_loss/val', 1 - self.ssim(combined_output, x_high))
        
        return g_loss

    def on_training_epoch_end(self):
        """Compute and log training metrics at the end of each epoch."""
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()

    def train_dataloader(self):
        """Return the training dataloader."""
        return self.train_loader

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.val_loader


class TrainPipeline(L.LightningModule):
    """Pipeline for training various models."""

    def __init__(self, config, train_loader, val_loader) -> None:
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        super().__init__()
        self.config = config

        # Initialize model
        self.model = MiniUnet(config['unet_layers'])
        
        # Load pre-trained weights if specified
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('Loaded successfully')
            
        # Initialize loss functions and metrics
        self.criterion = TorchLoss()
        self.mse_loss = MeanSquaredError()
        self.ssim = SSIM(
            data_range=1,
            size_average=True,
            channel=1,
            nonnegative_ssim=True,
            spatial_dims=3
        )
        
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_training_steps = len(self.train_loader)

        self.save_hyperparameters(config)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Configure optimizer
        if self.config['optimizer'] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9,
                nesterov=True,
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.config['optimizer']}")

        # Configure scheduler
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
        """Return the training dataloader."""
        return self.train_loader

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        x, y = batch
        out = self.model(x)
        mask = x == 0
        out = out * mask + x
        
        # Calculate loss
        loss = self.criterion(out, y)
        self.log("Loss/train", loss, prog_bar=True)
        
        # Update metrics
        self.train_metrics.update(out, y)
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(out) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        )
        
        # Log metrics
        self.log('R/train', R_factor)
        self.log('SSIM_loss/train', 1 - self.ssim(out, y))
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        x, y = batch
        out = self.model(x)
        mask = x == 0
        out = out * mask + x
        
        # Calculate loss
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        
        # Update metrics
        self.valid_metrics.update(out, y)
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(out) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        )
        
        # Log metrics
        self.log('R/val', R_factor)
        self.log('SSIM_loss/val', 1 - self.ssim(out, y))
 
    def on_training_epoch_end(self):
        """Compute and log training metrics at the end of each epoch."""
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of each epoch."""
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()


class TestPipeline(L.LightningModule):
    """Pipeline for testing models."""

    def __init__(self, config):
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Initialize model and metrics
        self.criterion = TorchLoss()
        self.model = MiniUnet(config['unet_layers'])
        self.ssim = SSIM(
            data_range=1,
            size_average=True,
            channel=1,
            nonnegative_ssim=True,
            spatial_dims=3
        )
        self.mse_loss = MeanSquaredError()
        
        # Load pre-trained weights
        state_dict = {}
        state_old = torch.load(config['weights'])['state_dict']
        for key in state_old.keys():
            key_new = key[6:]
            state_dict[key_new] = state_old[key]
        self.model.load_state_dict(state_dict, strict=True)
        print('Loaded successfully')
        
        # Initialize storage for test results
        self.test_outputs = []
        self.test_outputs_pre1 = []
        self.test_outputs_pre2 = []
        self.ssims = []
        self.mses = []
        self.R_factors = []
        
        # Initialize laue type dictionaries if post2 is enabled
        if self.config.get('enable_post2', False):
            laue_types = {
                'romb': {'h': [0, 16], 'k': [0, 21], 'l': [0, 28]},
                'clin': {'h': [-13, 12], 'k': [0, 17], 'l': [0, 22]},
                'all': {'h': [-16, 16], 'k': [-14, 21], 'l': [0, 28]}
            }
            hkl_minmax = laue_types[self.config['laue']]
            dics = {'h': {}, 'k': {}, 'l': {}}
            for letter in 'hkl':
                for i in range(hkl_minmax[letter][1] - hkl_minmax[letter][0] + 1):
                    dics[letter][hkl_minmax[letter][0] + i] = i
            self.h2ind, self.k2ind, self.l2ind = dics['h'], dics['k'], dics['l']

    def postprocessing1(self, low, recon):
        """Apply first postprocessing step."""
        mask = low == 0
        return recon * mask + ~mask * low

    def postprocessing2(self, recon, groups):
        """Apply second postprocessing step if enabled."""
        if not self.config.get('enable_post2', False):
            return recon
            
        for idx in range(recon.shape[0]):
            group = ''.join(groups[idx].split()[:-2])
            ms = miller.build_set(
                crystal_symmetry=crystal.symmetry(
                    space_group_symbol=group,
                    unit_cell=(25, 25, 25, 90, 90, 90)
                ),
                anomalous_flag=False,
                d_min=0.8
            )
            ms_base = ms.customized_copy(
                space_group_info=ms.space_group().build_derived_point_group().info()
            )
            ms_all = ms_base.complete_set()
            sys_abs = ms_all.lone_set(other=ms_base)
            sys_abs_list = list(sys_abs.indices())
            
            for ind in sys_abs_list:
                h, k, l = ind
                if h in self.h2ind.keys() and k in self.k2ind.keys() and l in self.l2ind.keys():
                    recon[idx, 0, self.h2ind[h], self.k2ind[k], self.l2ind[l]] = 0
        return recon
        
    def sync_across_gpus(self, tensors):
        """Synchronize tensors across GPUs."""
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_step(self, batch):
        """Execute a single test step."""
        x, y = batch
        out = self.model(x)
        
        # Calculate initial loss
        self.test_outputs.append(self.criterion(y, out).cpu().numpy())
        
        # Apply first postprocessing step
        recon = self.postprocessing1(x, out)
        self.test_outputs_pre1.append(self.criterion(y, recon).cpu().numpy())
        
        # Apply second postprocessing step if enabled
        if self.config.get('enable_post2', False):
            recon = self.postprocessing2(recon, gr)
            self.test_outputs_pre2.append(self.criterion(y, recon).cpu().numpy())
            
        # Calculate metrics
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(recon) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        ).cpu().numpy()
        
        self.R_factors.append(R_factor)
        self.mses.append(self.mse_loss(y, recon).cpu().numpy())
        self.ssims.append(self.ssim(y, recon).cpu().numpy())

    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of the epoch."""
        print(f'Without preprocessing: {np.mean(self.test_outputs)}')
        print(f'With preprocessing 1: {np.mean(self.test_outputs_pre1)}')
        if self.config.get('enable_post2', False):
            print(f'With preprocessing 1 + 2: {np.mean(self.test_outputs_pre2)}')
        print(f'MSE: {np.mean(self.mses)}')
        print(f'SSIM: {np.mean(self.ssims)}')
        print(f'R-factor: {np.mean(self.R_factors)}')


class XRDTransformerTestPipeline(L.LightningModule):
    """Pipeline for testing XRD Transformer model."""

    def __init__(self, config):
        """Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Initialize XRDTransformer model
        self.model = XRDTransformer(
            input_shape=(26, 18, 23),
            embed_dim=128,
            depth=5,
            num_heads=4,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            embedding_type='onehot'
        )
        
        # Load pre-trained weights
        if config['weights'] is not None:
            state_dict = {}
            state_old = torch.load(config['weights'])['state_dict']
            for key in state_old.keys():
                key_new = key[6:]
                state_dict[key_new] = state_old[key]
            self.model.load_state_dict(state_dict, strict=True)
            print('Loaded successfully')
        
        # Initialize metrics
        self.criterion = TorchLoss()
        self.mse_loss = MeanSquaredError()
        self.ssim = SSIM(
            data_range=1,
            size_average=True,
            channel=1,
            nonnegative_ssim=True,
            spatial_dims=3
        )
        
        # Initialize storage for test results
        self.test_outputs = []
        self.test_outputs_pre1 = []
        self.ssims = []
        self.mses = []
        self.R_factors = []

    def postprocessing1(self, low, recon):
        """Apply postprocessing step."""
        mask = low == 0
        return recon * mask + ~mask * low

    def test_step(self, batch, batch_idx):
        """Execute a single test step."""
        x, y = batch
        
        # Forward pass through model
        out = self.model(x)
        
        # Calculate initial loss
        self.test_outputs.append(self.criterion(y, out).cpu().numpy())
        
        # Apply postprocessing
        recon = self.postprocessing1(x, out)
        self.test_outputs_pre1.append(self.criterion(y, recon).cpu().numpy())
        
        # Calculate metrics
        R_factor = torch.mean(
            torch.sum(torch.abs(torch.abs(recon) - torch.abs(y)), axis=(1, 2, 3, 4)) /
            torch.sum(torch.abs(y), axis=(1, 2, 3, 4))
        ).cpu().numpy()
        
        self.R_factors.append(R_factor)
        self.mses.append(self.mse_loss(y, recon).cpu().numpy())
        self.ssims.append(self.ssim(y, recon).cpu().numpy())

    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of the epoch."""
        print(f'Raw output loss: {np.mean(self.test_outputs)}')
        print(f'Post-processed loss: {np.mean(self.test_outputs_pre1)}')
        print(f'MSE: {np.mean(self.mses)}')
        print(f'SSIM: {np.mean(self.ssims)}')
        print(f'R-factor: {np.mean(self.R_factors)}')