import torch
import torchvision
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from models.fomm.model import GeneratorFullModel, DiscriminatorFullModel
from utils.visualizer import Visualizer


class RunnerFOMM(pl.LightningModule):
    
    def __init__(self, cfg, model_pack, start_epoch):
        super().__init__()
        self.cfg = cfg
        self.params = self.cfg.train_params
        self.start_epoch = start_epoch
        
        # Set-up model
        self.generator = model_pack['generator']
        self.discriminator = model_pack['discriminator']
        self.kp_detector = model_pack['kp_detector']
        self.generator_full = GeneratorFullModel(self.kp_detector, self.generator, self.discriminator, self.params, self.device)
        self.discriminator_full = DiscriminatorFullModel(self.kp_detector, self.generator, self.discriminator, self.params, self.device)
        
        # Set-up visualizer
        self.visualizer = Visualizer()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
    
    
    def training_step(self, batch, batch_idx):
        
        # Fetch optimizer
        g_opt, d_opt, kp_opt = self.optimizers()
        
        # Model forwarding 
        losses_generator, generated = self.generator_full(batch)
        
        loss_values = [val.mean() for val in losses_generator.values()]
        loss = sum(loss_values)

        self.manual_backward(loss)
        g_opt.step()
        g_opt.zero_grad()
        kp_opt.step()
        kp_opt.zero_grad()
        
        # Discriminator
        if self.params.loss_weights.generator_gan != 0:
            d_opt.zero_grad()

            losses_discriminator = self.discriminator_full(batch, generated)
            loss_values = [val.mean() for val in losses_discriminator.values()]
            loss = sum(loss_values)

            self.manual_backward(loss)
            d_opt.step()
            d_opt.zero_grad()
        else:
            losses_discriminator = {}
        
        # Logging
        losses_generator.update(losses_discriminator)
        losses = {key: value.mean().detach().data.cpu() for key, value in losses_generator.items()}
    
        for k, v in losses.items():
            self.log(k, v, on_epoch=True)
    
        if batch_idx < 5:
            image = self.visualizer.visualize_fomm(batch['driving'], batch['source'], generated)
            tensor_image = torchvision.transforms.ToTensor()(image)            
            self.logger.experiment.add_image(f'image-{batch_idx}', tensor_image, self.global_step)

        return {}        
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def training_epoch_end(self, outputs) -> None:        
        # Scheduler update
        g_sch, d_sch, kp_sch = self.lr_schedulers()
        g_sch.step()
        d_sch.step()
        kp_sch.step()        
    
    def validation_epoch_end(self, outputs) -> None:
        pass
    
    def configure_optimizers(self):
        optimizer_generator = optim.Adam(self.generator.parameters(), lr=self.params.lr_generator, betas=(0.5, 0.999))
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.params.lr_discriminator, betas=(0.5, 0.999))
        optimizer_kp_detector = optim.Adam(self.kp_detector.parameters(), lr=self.params.lr_kp_detector, betas=(0.5, 0.999))
        scheduler_generator = MultiStepLR(optimizer_generator, self.params.epoch_milestones, gamma=0.1, last_epoch=self.start_epoch - 1)
        scheduler_discriminator = MultiStepLR(optimizer_discriminator, self.params.epoch_milestones, gamma=0.1, last_epoch=self.start_epoch - 1)
        scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, self.params.epoch_milestones,  gamma=0.1, last_epoch=-1 + self.start_epoch * (self.params.lr_kp_detector != 0))
        return [optimizer_generator, optimizer_discriminator, optimizer_kp_detector],\
               [scheduler_generator, scheduler_discriminator, scheduler_kp_detector]
    
    def custom_batch_test(self, batch) -> dict:
        tested_result = {}
        
        with torch.no_grad():
            kp_source = self.kp_detector(batch['source'])
            kp_driving = self.kp_detector(batch['driving'])
            
            generated = self.generator(batch['source'], kp_source=kp_source, kp_driving=kp_driving)
            
        # make output
        tested_result['source'] = batch['source']
        tested_result['driving'] = batch['driving']
        tested_result['generated'] = generated
        
        return tested_result