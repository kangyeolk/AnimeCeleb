import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functions import convert_flow_to_deformation, warp_image
from models.pirender.base_function import LayerNorm2d, ADAINHourglass, FineEncoder, FineDecoder

class FaceGenerator(nn.Module):
    def __init__(self, mapping_net, warpping_net, editing_net, common):  
        super(FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.warpping_net = WarpingNet(**warpping_net, **common)
        self.editing_net = EditingNet(**editing_net, **common)
 
    def forward(self, input_image, driving_source, stage=None):
        if stage == 'warp':
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
        else:
            descriptor = self.mapping_net(driving_source)
            output = self.warpping_net(input_image, descriptor)
            output['fake_image'] = self.editing_net(input_image, output['warp_image'], descriptor)
        return output

class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer, multi_frame=True, align_fc=False):
        super( MappingNet, self).__init__()

        self.layer = layer
        self.multi_frame = multi_frame
        self.align_fc = align_fc
        nonlinearity = nn.LeakyReLU(0.1)
        
        if align_fc:
            self.alignA = nn.Linear(20, coeff_nc, bias=True)
            self.alignV = nn.Linear(70, coeff_nc, bias=True)

        if multi_frame:
            self.first = nn.Sequential(
                torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

            for i in range(layer):
                net = nn.Sequential(nonlinearity,
                    torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
                setattr(self, 'encoder' + str(i), net)   
            
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.first = nn.Sequential(
                torch.nn.Linear(coeff_nc, descriptor_nc, bias=True))

            for i in range(layer):
                net = nn.Sequential(nonlinearity,
                    torch.nn.Linear(descriptor_nc, descriptor_nc))
                setattr(self, 'encoder' + str(i), net)
        
        self.output_nc = descriptor_nc

    def forward(self, input_pose):
        if self.align_fc:
            if input_pose.size(-1) == 20:
                input_pose = self.alignA(input_pose)
            elif input_pose.size(-1) == 70:
                input_pose = self.alignV(input_pose)
        
        out = self.first(input_pose)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            if self.multi_frame:
                out = model(out) + out[:,:,3:-3]
            else:
                out = model(out) + out
        if self.multi_frame:
            out = self.pooling(out)
        return out   

class WarpingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        base_nc, 
        max_nc, 
        encoder_layer, 
        decoder_layer, 
        use_spect
        ):
        super( WarpingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'nonlinearity':nonlinearity, 'use_spect':use_spect}

        self.descriptor_nc = descriptor_nc 
        self.hourglass = ADAINHourglass(image_nc, self.descriptor_nc, base_nc,
                                       max_nc, encoder_layer, decoder_layer, **kwargs)

        self.flow_out = nn.Sequential(norm_layer(self.hourglass.output_nc), 
                                      nonlinearity,
                                      nn.Conv2d(self.hourglass.output_nc, 2, kernel_size=7, stride=1, padding=3))

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, descriptor):
        final_output={}
        output = self.hourglass(input_image, descriptor)
        final_output['flow_field'] = self.flow_out(output)        
        deformation = convert_flow_to_deformation(final_output['flow_field'])

        
        final_output['warp_image'] = warp_image(input_image, deformation)
        return final_output


class EditingNet(nn.Module):
    def __init__(
        self, 
        image_nc, 
        descriptor_nc, 
        layer, 
        base_nc, 
        max_nc, 
        num_res_blocks, 
        use_spect):  
        super(EditingNet, self).__init__()

        nonlinearity = nn.LeakyReLU(0.1)
        norm_layer = functools.partial(LayerNorm2d, affine=True) 
        kwargs = {'norm_layer':norm_layer, 'nonlinearity':nonlinearity, 'use_spect':use_spect}
        self.descriptor_nc = descriptor_nc

        # encoder part
        self.encoder = FineEncoder(image_nc*2, base_nc, max_nc, layer, **kwargs)
        self.decoder = FineDecoder(image_nc, self.descriptor_nc, base_nc, max_nc, layer, num_res_blocks, **kwargs)

    def forward(self, input_image, warp_image, descriptor):
        x = torch.cat([input_image, warp_image], 1)
        x = self.encoder(x)
        gen_image = self.decoder(x, descriptor)
        return gen_image