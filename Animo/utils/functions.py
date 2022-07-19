import os
import yaml
from easydict import EasyDict as edict
import torch
from torch.nn import init
import matplotlib.pyplot as plt

def load_config_file(config_path, return_edict=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return edict(cfg) if return_edict else cfg

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(0., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor

def tensor2image_with_denorm(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def find_ext_recursively(folder, extensions: tuple):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(extensions):
                matches.append(os.path.join(root, filename))
    return matches

def load_ckpt(cfg, model_pack, ckpt_data):
    # load ckpt
    # model_pack = ckpt_data
    return model_pack

def weights_init(init_type='normal', gain=0.02, bias=None):
    r"""Initialize weights in the network.

    Args:
        init_type (str): The name of the initialization scheme.
        gain (float): The parameter that is required for the initialization
            scheme.
        bias (object): If not ``None``, specifies the initialization parameter
            for bias.

    Returns:
        (obj): init function to be applied.
    """

    def init_func(m):
        r"""Init function

        Args:
            m: module to be weight initialized.
        """
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (
                class_name.find('Conv') != -1 or
                class_name.find('Linear') != -1 or
                class_name.find('Embedding') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    'initialization method [%s] is '
                    'not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                if bias is not None:
                    bias_type = getattr(bias, 'type', 'normal')
                    if bias_type == 'normal':
                        bias_gain = getattr(bias, 'gain', 0.5)
                        init.normal_(m.bias.data, 0.0, bias_gain)
                    else:
                        raise NotImplementedError(
                            'initialization method [%s] is '
                            'not implemented' % bias_type)
                else:
                    init.constant_(m.bias.data, 0.0)
    return init_func


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b,c,h,w = flow.shape
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation


def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """    
    b,c,h,w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed    

    
def warp_image(source_image, deformation):
    r"""warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """ 
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation, padding_mode='border')


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)




