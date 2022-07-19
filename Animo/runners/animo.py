import math
import os
import json
import numpy as np

import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import StepLR

from losses.perceptual import PerceptualLoss
from utils.visualizer import Visualizer


class RunnerAnimo(pl.LightningModule):
    
    def __init__(self, cfg, model_pack, start_epoch):
        super().__init__()
        self.cfg = cfg
        self.params = self.cfg.train_params
        self.start_epoch = start_epoch
        self.batch_size = self.params.batch_size

        # Set-up model
        self.mapping_net = model_pack['mapping_net_shared']
        self.warpping_netA = model_pack['warpping_net_anime']
        self.editing_netA = model_pack['editing_net_anime']
        self.warpping_netV = model_pack['warpping_net_vox']
        self.editing_netV = model_pack['editing_net_vox']
        
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
            
        # Set-up basis        
        if os.path.isfile('../../../data/pose_coeffs.npy'):
            full_coeffs = torch.from_numpy(np.load('../../../data/pose_coeffs.npy'))
        elif os.path.isfile('../data/pose_coeffs.npy'): 
            full_coeffs = torch.from_numpy(np.load('../data/pose_coeffs.npy'))
        elif os.path.isfile('./data/pose_coeffs.npy'): 
            full_coeffs = torch.from_numpy(np.load('./data/pose_coeffs.npy'))
        self.exp_coeffs = full_coeffs[:, 80:144]                                  # (17 , 64)
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
        # Training Stage
        self.training_stage = 'warp'
    
    def convert_angles_to_3dmm(self, rpy):    
        # [Pose vec] roll, pitch, yaw (angle) --> [3DMM] pitch, yaw, roll (radian)
        # rpy: torch.Tensor, (B, 3)
        radian_3dmm = (math.pi / 180) * rpy
        radian_3dmm = radian_3dmm[:, [0, 2, 1]]
        return radian_3dmm
    
    def convert_poses_to_3dmm(self, pose):
        # pose: torch.Tensor, (B, 20)
        exp_pose = pose[:, :17]
        rpy = pose[:, 17:]
        exp_coeffs = self.exp_coeffs.to(exp_pose.device)
        
        radian_3dmm = self.convert_angles_to_3dmm(rpy)
        # (B, 17) * (17, 64) = (B, 64)
        exp_3dmm = torch.mm(exp_pose, exp_coeffs)
        trans_params = torch.zeros_like(radian_3dmm).to(exp_3dmm.device)
        
        transformed_3dmm = torch.cat([exp_3dmm, radian_3dmm, trans_params], -1) # (B, 70)
        return transformed_3dmm
    
    def forward_until_warpnet(self, input_image, input_pose, warp_domain='A'):
        # Mapping + warping
        descriptor = self.mapping_net(input_pose)
        if warp_domain == 'A':
            output = self.warpping_netA(input_image, descriptor)
        elif warp_domain == 'V':
            output = self.warpping_netV(input_image, descriptor)

        return descriptor, output
    
    def training_step(self, batch, batch_idx):
        loss = {}

        # Fetch optimizer
        optimizer_G = self.optimizers()    
        
        # Data Load (A: Anime / V: Vox)
        input_imageA = torch.cat((batch['anime']['source'], batch['anime']['driving']), 0)
        input_poseA = torch.cat((batch['anime']['driving_pose'], batch['anime']['source_pose']), 0)
        input_poseA = self.convert_poses_to_3dmm(input_poseA)
        gt_imageA = torch.cat((batch['anime']['driving'], batch['anime']['source']), 0)
        
        input_imageV = torch.cat((batch['vox']['source'], batch['vox']['driving']), 0)
        input_poseV = torch.cat((batch['vox']['driving_pose'], batch['vox']['source_pose']), 0)
        gt_imageV = torch.cat((batch['vox']['driving'], batch['vox']['source']), 0)
        
        
        # Model forwarding
        # Self-training (Anime)
        descriptor_edit_A, outputAA = self.forward_until_warpnet(input_imageA, input_poseA, warp_domain='A')
        if self.training_stage == 'gen':            
            outputAA['fake_image'] = self.editing_netA(input_imageA, outputAA['flow_field'], descriptor_edit_A)

        # Self-training (Vox)
        descriptor_edit_V, outputVV = self.forward_until_warpnet(input_imageV, input_poseV, warp_domain='V')    
        if self.training_stage == 'gen':
            outputVV['fake_image'] = self.editing_netV(input_imageV, outputVV['flow_field'], descriptor_edit_V)

       
        loss['warpAA'] = self.weight_perceptual_warp * self.perceptual_warp(outputAA['warp_image'], gt_imageA)            
        
        if self.training_stage == 'gen':
            loss['genAA'] = self.weight_perceptual_final * self.perceptual_final(outputAA['fake_image'], gt_imageA)
            loss['totalAA'] = loss['warpAA'] + loss['genAA']
        else:
            loss['totalAA'] = loss['warpAA']
        
        loss['warpVV'] = self.weight_perceptual_warp * self.perceptual_warp(outputVV['warp_image'], gt_imageV)
                
        if self.training_stage == 'gen':
            loss['genVV'] = self.weight_perceptual_final * self.perceptual_final(outputVV['fake_image'], gt_imageV)
            loss['totalVV'] = loss['warpVV'] + loss['genVV']
        else:
            loss['totalVV'] = loss['warpVV']    
        loss['total'] = loss['totalAA'] + loss['totalVV']
        
        # Backward
        optimizer_G.zero_grad()
        self.manual_backward(loss['total'])
        optimizer_G.step()
        
        for k, v in loss.items():
            self.log(k, v.mean().detach().data.cpu(), on_epoch=True)

        return {}        
    
    def check_directory(self, path):
        # shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    
    def save_image(self, x, ncol, filename):
        vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
    
    def save_json(self, json_file, filename):
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                json.dump(json_file, f, indent=4, sort_keys=False)
        else:
            with open(filename, 'a') as f:
                json.dump(json_file, f, indent=4, sort_keys=False)

    
    def validation_step(self, batch, batch_idx):
        # Switching the training stage
        if self.current_epoch >= 99:
            self.training_stage = 'gen'
        
        # return # For Only Discriminator Training
        
        log_images = (batch_idx < 10)
                        
        with torch.no_grad():
            
            if log_images:
                # Data Load (A: Anime / V: Vox)
                input_imageA = batch['anime']['source']
                input_poseA = batch['anime']['driving_pose']
                input_poseA = self.convert_poses_to_3dmm(input_poseA)
                gt_imageA = batch['anime']['driving']
                
                input_imageV = batch['vox']['source']
                input_poseV = batch['vox']['driving_pose']
                gt_imageV = batch['vox']['driving']
                
                # Model forwarding 
                # Self-training (Anime)                
                descriptor_edit, outputAA = self.forward_until_warpnet(input_imageA, input_poseA, warp_domain='A')
                outputAA['fake_image'] = self.editing_netA(input_imageA, outputAA['flow_field'], descriptor_edit)

                # Self-training (Vox)
                descriptor_edit, outputVV = self.forward_until_warpnet(input_imageV, input_poseV, warp_domain='V')
                outputVV['fake_image'] = self.editing_netV(input_imageV, outputVV['flow_field'], descriptor_edit)

                # Cross-training (Vox -> Anime)
                input_poseVA = torch.cat((input_poseV[:, :67], input_poseA[:, 67:]), dim=-1)
                descriptor_edit, outputVA = self.forward_until_warpnet(input_imageA, input_poseVA, warp_domain='A')                
                outputVA['fake_image'] = self.editing_netA(input_imageA, outputVA['flow_field'], descriptor_edit)
                
                # Cross-training (Anime -> Vox)
                input_poseAV = torch.cat((input_poseA[:, :67], input_poseV[:, 67:]), dim=-1)
                descriptor_edit, outputAV = self.forward_until_warpnet(input_imageV, input_poseAV, warp_domain='V')                                
                outputAV['fake_image'] = self.editing_netV(input_imageV, outputAV['flow_field'], descriptor_edit)
                
                if log_images:
                    outp = dict()
                    outp['sourceV'] = input_imageV
                    outp['sourceA'] = input_imageA
                    outp['drivingV'] = gt_imageV
                    outp['drivingA'] = gt_imageA
                    outp['warpAA'] = outputAA['warp_image']
                    outp['fakeAA'] = outputAA['fake_image']
                    outp['warpVV'] = outputVV['warp_image']
                    outp['fakeVV'] = outputVV['fake_image']
                    outp['warpAV'] = outputAV['warp_image']
                    outp['fakeAV'] = outputAV['fake_image']
                    outp['warpVA'] = outputVA['warp_image']
                    outp['fakeVA'] = outputVA['fake_image']
                    outp['flowAA'] = outputAA['flow_field']
                    outp['flowVV'] = outputVV['flow_field']
                    outp['flowAV'] = outputAV['flow_field']
                    outp['flowVA'] = outputVA['flow_field']

                    image_output = self.visualizer.visualize_basis(outp)                
                    self.logger.experiment.add_image(f'image-{batch_idx}', image_output, self.current_epoch)

    def training_epoch_end(self, outputs) -> None:
        # Scheduler update
        scheduler = self.lr_schedulers()
        scheduler.step()
        
        # Switching the training stage
        if self.current_epoch >= 99:
            self.training_stage = 'gen'

    def validation_epoch_end(self, outputs) -> None:
        pass
    
    def configure_optimizers(self):
        params_to_opt_G = list()
        params_to_opt_G += list(self.mapping_net.parameters())
        params_to_opt_G += list(self.warpping_netA.parameters())
        params_to_opt_G += list(self.editing_netA.parameters())
        params_to_opt_G += list(self.warpping_netV.parameters())
        params_to_opt_G += list(self.editing_netV.parameters())

        optimizer_G = optim.Adam(params_to_opt_G, lr=self.params.lr_generator, betas=(0.5, 0.999))
        scheduler_G = StepLR(optimizer_G, step_size=150, gamma=0.2)

        return [optimizer_G], [scheduler_G]

    def custom_batch_test(self, batch, dsrc, ddrv) -> dict:
        tested_result = {}
        
        # Fetch
        source = batch['anime']['source'] if dsrc == 'anime' else batch['vox']['source']
        driving = batch['anime']['driving'] if ddrv == 'anime' else batch['vox']['driving'] 
        if ddrv == 'anime':
            pose = self.convert_poses_to_3dmm(batch['anime']['driving_pose'])
        else:
            pose = batch['vox']['driving_pose']
            
        # Forwarding
        if dsrc == 'anime':
            descriptor_edit, output = self.forward_until_warpnet(source, pose, warp_domain='A')
            fake_image, feats = self.editing_netA.forward_with_feats(source, output['flow_field'], descriptor_edit, None)
        else:
            descriptor_edit, output = self.forward_until_warpnet(source, pose, warp_domain='V')
            fake_image, feats = self.editing_netV.forward_with_feats(source, output['flow_field'], descriptor_edit, None)
        
        # Append product
        tested_result['source'] = source
        tested_result['driving'] = driving        
        tested_result['fake_image'] = fake_image
        tested_result['feats'] = feats
        
        return tested_result
