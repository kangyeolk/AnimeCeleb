from models.fomm.generator import OcclusionAwareGenerator
from models.fomm.discriminator import MultiScaleDiscriminator
from models.fomm.keypoint_detector import KPDetector

from utils.functions import weights_init


def get_model_pack(cfg):
    model_pack = {}
    
    if cfg.method == 'fomm':
        generator = OcclusionAwareGenerator(**cfg['model_params']['generator_params'],
                                            **cfg['model_params']['common_params'])
        discriminator = MultiScaleDiscriminator(**cfg['model_params']['discriminator_params'],
                                                **cfg['model_params']['common_params'])
        kp_detector = KPDetector(**cfg['model_params']['kp_detector_params'],
                                **cfg['model_params']['common_params'])

        model_pack['generator'] = generator
        model_pack['discriminator'] = discriminator
        model_pack['kp_detector'] = kp_detector
    
    elif cfg.method in ['pirender', 'pirender_mlp']:
        from models.pirender.model import FaceGenerator

        generator = FaceGenerator(mapping_net=cfg['model_params']['mappingnet_params'], 
                                  warpping_net=cfg['model_params']['warpingnet_params'], 
                                  editing_net=cfg['model_params']['editingnet_params'], 
                                  common=cfg['model_params']['common'])
        generator_ema = FaceGenerator(mapping_net=cfg['model_params']['mappingnet_params'], 
                                      warpping_net=cfg['model_params']['warpingnet_params'], 
                                      editing_net=cfg['model_params']['editingnet_params'], 
                                      common=cfg['model_params']['common'])
        
        # Weight Initialization
        generator.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        accumulate(generator_ema, generator, 0)

        model_pack['generator'] = generator
        model_pack['generator_ema'] = generator_ema
        
    elif cfg.method == 'animo':
        from models.animo.model import MappingNet, WarpingNet, EditingNet
        mappingnet_params = cfg['model_params']['mappingnet_params']
        warpingnet_params = cfg['model_params']['warpingnet_params']
        editingnet_params = cfg['model_params']['editingnet_params']
        common = cfg['model_params']['common']

        mapping_net_shared = MappingNet(**mappingnet_params)
        
        warpping_net_anime = WarpingNet(**warpingnet_params, **common)        
        warpping_net_real = WarpingNet(**warpingnet_params, **common)        
        editing_net_anime = EditingNet(**editingnet_params, **common)
        editing_net_real = EditingNet(**editingnet_params, **common)
        
        # Weight Initialization
        mapping_net_shared.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        warpping_net_anime.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        warpping_net_real.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        editing_net_anime.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        editing_net_real.apply(weights_init(init_type='normal', gain=0.02, bias=None))
        
        model_pack['mapping_net_shared'] = mapping_net_shared
        model_pack['warpping_net_anime'] = warpping_net_anime
        model_pack['editing_net_anime'] = editing_net_anime
        model_pack['warpping_net_vox'] = warpping_net_real
        model_pack['editing_net_vox'] = editing_net_real

    return model_pack


def weights_update(state_dict, ckpt_name):    
    # Get target contained layer
    pretrained_dict = {k: v for k, v in state_dict.items() if ckpt_name in k}
    # Remove target layer
    ckpt_name_dot = ckpt_name + '.'
    pretrained_dict = {k.replace(ckpt_name_dot, ''): v for k, v in pretrained_dict.items()}    
    return pretrained_dict

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

