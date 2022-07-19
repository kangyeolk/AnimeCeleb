import math
import os 
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch import optim
from torch.optim.lr_scheduler import StepLR

from losses.perceptual import PerceptualLoss
from utils.visualizer import Visualizer


class RunnerPIRender(pl.LightningModule):
    
    def __init__(self, cfg, model_pack, start_epoch):
        super().__init__()
        self.cfg = cfg
        self.params = self.cfg.train_params
        self.start_epoch = start_epoch
        
        # Set-up model
        self.generator = model_pack['generator']

        # Set-up visualizer
        self.visualizer = Visualizer()

        # Set-up loss
        self.perceptual_warp = PerceptualLoss(network=self.params['vgg_param_warp']['network'], 
                                              layers=self.params['vgg_param_warp']['layers'],
                                              num_scales=self.params['vgg_param_warp']['num_scales'], 
                                              use_style_loss=self.params['vgg_param_warp']['use_style_loss'], 
                                              weight_style_to_perceptual=0)
        self.perceptual_final = PerceptualLoss(network=self.params['vgg_param_final']['network'], 
                                               layers=self.params['vgg_param_final']['layers'],
                                               num_scales=self.params['vgg_param_final']['num_scales'], 
                                               use_style_loss=self.params['vgg_param_final']['use_style_loss'], 
                                               weight_style_to_perceptual=self.params['vgg_param_final']['style_to_perceptual'])
        self.weight_perceptual_warp = self.params['loss_weights']['weight_perceptual_warp']
        self.weight_perceptual_final = self.params['loss_weights']['weight_perceptual_final']
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
        # Training Stage
        self.training_stage = 'warp'
    
    def training_step(self, batch, batch_idx):
        loss = {}

        # Fetch optimizer
        optimizer = self.optimizers()
        
        # Model forwarding
        source_image, target_image = batch['source'], batch['driving']
        source_pose, target_pose = batch['source_pose'], batch['driving_pose']
        
        input_image = torch.cat((source_image, target_image), 0)
        input_pose = torch.cat((target_pose, source_pose), 0)
        gt_image = torch.cat((target_image, source_image), 0)

        output_dict = self.generator(input_image, input_pose, self.training_stage)

        if self.training_stage == 'gen':
            fake_img = output_dict['fake_image']
            warp_img = output_dict['warp_image']
            loss['warp'] = self.weight_perceptual_warp * self.perceptual_warp(warp_img, gt_image)
            loss['gen'] = self.weight_perceptual_final * self.perceptual_final(fake_img, gt_image)
            loss['total'] = loss['warp'] + loss['gen']
        else:
            warp_img = output_dict['warp_image']
            loss['total'] = self.weight_perceptual_warp * self.perceptual_warp(warp_img, gt_image)
        
        # Backward
        self.manual_backward(loss['total'])
        optimizer.step()
        optimizer.zero_grad()
                
        for k, v in loss.items():
            self.log(k, v.mean().detach().data.cpu(), on_epoch=True)

        return {}        
    
    def validation_step(self, batch, batch_idx):
        if batch_idx < 10:
            with torch.no_grad():
                # Model forwarding
                source_image, target_image = batch['source'], batch['driving']
                source_pose, target_pose = batch['source_pose'], batch['driving_pose']
                
                input_image = torch.cat((source_image, target_image), 0)
                input_pose = torch.cat((target_pose, source_pose), 0)

                output_dict = self.generator(input_image, input_pose, self.training_stage)
                
                if self.training_stage == 'gen':
                    image = self.visualizer.visualize_pirender(target_image, source_image, output_dict['fake_image'][:self.params['batch_size']])
                else:
                    image = self.visualizer.visualize_pirender(target_image, source_image, output_dict['warp_image'][:self.params['batch_size']])
                tensor_image = torchvision.transforms.ToTensor()(image)            
                self.logger.experiment.add_image(f'image-{batch_idx}', tensor_image, self.global_step)
        else:
            pass
    
    def training_epoch_end(self, outputs) -> None:  
        # Scheduler update
        scheduler = self.lr_schedulers()
        scheduler.step()
        
        # Switching the training stage
        if self.current_epoch == 99:
            self.training_stage = 'gen'
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.generator.parameters(), lr=self.params.lr_generator, betas=(0.5, 0.999))
        scheduler = StepLR(optimizer, step_size=150, gamma=0.2)
        return [optimizer], [scheduler]
    
    def custom_batch_test(self, batch) -> dict:
        tested_result = {}
        
        with torch.no_grad():
            source_image, target_image = batch['source'], batch['driving']
            source_pose, target_pose = batch['source_pose'], batch['driving_pose']
            
            output_dict = self.generator(source_image, target_pose, 'gen')
            
        # make output
        tested_result['source'] = batch['source']
        tested_result['driving'] = batch['driving']
        tested_result['generated'] = output_dict
        
        return tested_result
        