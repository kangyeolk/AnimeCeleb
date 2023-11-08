import torch
import sys
import imageio
import matplotlib.pyplot as plt
import json
sys.path.insert(0, '../Animo')
from data import get_dm
from runners import get_runner, get_runner_class
from models import get_model_pack
from utils.functions import *
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import numpy as np
from skimage import io, img_as_float32
from skimage.transform import resize
from datetime import datetime
from tqdm import tqdm
from scipy.io import loadmat
import math
import os
from torchvision.utils import make_grid
from torchvision.utils import save_image
from util import loop_iterable, set_requires_grad, GrayscaleToRgb, one_vox_deca_pose, convert_deca_to_angles, get_anime_basis
import argparse
from PIL import Image
import glob

## hyperparameter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument
    
    ## required
    parser.add('--path_pose', type=str, default='./example/vox_deca')
    parser.add('--anime_basis_path', type=str, default='./example/basis.png')
    parser.add('--decapath', type=str, default = '../../DECA')
    parser.add('--coeff_path', type=str, default = './pretrained/transform_matrix_08_30.npy')
    parser.add('--exp_path', type=str, default=f'../../Animation-talking-head/outputs/03-38-47_pirenderer')
    parser.add('--EDTN_path', type=str, default=f'./pretrained/EDTN.pt')
    
    parser.add('--gpu_id', type=int, default=0)
    parser.add('--num_epochs', type=int, default = 100)
    parser.add('--batch_size', type=int, default = 512)
    parser.add('--num_workers', type=int, default = 4)

    parser.add('--exp_cfg', type=str, default = './config/config_EDTN.yaml')
    parser.add('--useTex', type=bool, default=False)
    parser.add('--rasterizer_type', type=str, default='standard')
    parser.add('--extractTex', type=bool, default=False)

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}')

    return args

class Generator2ani(nn.Module):
    def __init__(self):
        super(Generator2ani, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(53, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 17),
            nn.Sigmoid(),
        )
    def forward(self, input):
        return self.main(input).view(-1, 1, 17)
    
def get_files(path):
    list_files = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            for f in files:
                fullpath = os.path.join(root, f)
                list_files.append(fullpath)
    return list_files

if __name__ == '__main__':

    args = get_args()
    
    now = datetime.now()

    ## run Ani-Pirenderer
    device = args.device
    # exp_id = '03-38-47_pirenderer' # AnimeCeleb Only
    # exp_cfg = f'../../Animation-talking-head/outputs/{exp_id}/.hydra/config.yaml'
    #exp_cfg=f'../configs/ours_basis.yaml' #checkpoint X
    cfg = load_config_file(args.exp_cfg, return_edict=True)
    cfg.train_params.batch_size = args.batch_size
    cfg.train_params.num_workers = args.num_workers
    cfg.dataset_folder = 'rotation'

    runner_class = get_runner_class(cfg)
    model_ckpts = find_ext_recursively(folder=args.exp_path, extensions=('.ckpt'))
    print(f'There are {len(model_ckpts)} checkpoints, we will use {model_ckpts[-1]} for inference!')
    map_location = device
    runner_test = runner_class.load_from_checkpoint(model_ckpts[-1], cfg=cfg, model_pack=get_model_pack(cfg), start_epoch=0, map_location=map_location)

    deca2ani = Generator2ani()
    deca2ani = deca2ani.to(device)
    deca2ani.eval()
    deca2ani.load_state_dict(torch.load(args.EDTN_path, map_location = device))

    ## run DECA
    sys.path.append(args.decapath)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from decalib.deca import DECA
    from decalib.datasets import datasets 
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from decalib.utils.tensor_cropper import transform_points


    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)

    ## functions
    dmm_videos, vox_poses = one_vox_deca_pose(args.path_pose)
    ani_videos = get_anime_basis(args.anime_basis_path)
    frame_shape=(256, 256, 3)

    if not os.path.exists(f'./results/{now}/'):
        os.mkdir(f'./results/{now}/')

    runner_test = runner_test.eval().to(device)

    for idx in tqdm(range(len(dmm_videos))):
        ## make fake
        ani_source = torch.from_numpy((ani_videos[0].transpose((2, 0, 1)))).unsqueeze(dim=0).to(device)
        vox_driving_exp = torch.from_numpy(vox_poses[idx]).unsqueeze(dim=0).to(device)
        dmm_driving_image = torch.from_numpy(dmm_videos[idx].transpose((2, 0, 1))).unsqueeze(dim=0).to(device)

        fake_full_pose = torch.cat((deca2ani(torch.cat((vox_driving_exp[:,100:150],vox_driving_exp[:,153:156]),dim=-1)).squeeze().unsqueeze(dim=0).squeeze(),
                convert_deca_to_angles(vox_driving_exp[:,150:153]).squeeze()),dim = -1).unsqueeze(dim=0)

        fake_full_pose[12:17] = fake_full_pose[12:17] * 1.5
        batches={}
        batches['source'] = ani_source 
        batches['source_pose'] = vox_driving_exp
        batches['driving'] = dmm_driving_image
        batches['driving_pose'] = fake_full_pose

        pred = runner_test.custom_batch_test(batches)

        ## viz_check
        pred_cpu=pred['generated']['fake_image'][0].cpu().detach()
        driving_img_cpu = dmm_driving_image[0].cpu()
        source = ani_source[0].cpu()

        pred_cpu = pred_cpu.unsqueeze(dim=0)
        driving_img_cpu = driving_img_cpu.unsqueeze(dim=0)
        source = source.unsqueeze(dim=0)

        images = torch.cat((source, driving_img_cpu, pred_cpu), dim=0) # , vox_driving_image_cpu
        grid = make_grid(images, nrow=3, normalize=True, scale_each=True)
        grid = grid.detach()
        # grid = grid.transpose(1, 2, 0)


        save_image(grid, f'./results/{now}/{idx}.png')

    
    
    image_dir = os.path.join(f'./results/{now}')
    output_dir = './results/'
    
    gif_config = {
        'loop':1, 
        'duration': 0.05 
    }
    
    images = [plt.imread(os.path.join(image_dir, x)) for x in os.listdir(image_dir)]
    images = [(img * 255).astype(np.uint8) for img in images]


    imageio.mimwrite(os.path.join(output_dir, f'{now}.gif'), 
                    images, 
                    format='gif', 
                    **gif_config 
                    )
    
    ### this method has some noise. So i choose the method above even if it takes some time.
    ### if you want faster, you can use below method.

    # frames = []
    # imgs = glob.glob(f'./results/{now}/*.png')

    # for i in imgs:
    #     new_frame = Image.open(i)
    #     frames.append(new_frame)

    # frames[0].save(f'./results/{now}.gif', format='GIF',
    #             append_images=frames[1:],
    #             save_all=True,
    #             duration=0.01,
    #             loop=0
    #             )
    
    print(f"gif saved.")

