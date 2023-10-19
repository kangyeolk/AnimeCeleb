
import torch
import sys
sys.path.insert(0, '../Animo')
from data import get_dm
from runners import get_runner, get_runner_class
from models import get_model_pack
from utils.functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import numpy as np
from skimage import io, img_as_float32
from skimage.transform import resize
import datetime
from tqdm import tqdm
from scipy.io import loadmat
import math
import os
import matplotlib.pyplot as plt

from util import set_requires_grad, landmark_loss, eye_dis, eyed_loss, lip_dis, lipd_loss, numpy2image, landmark_loss_tensor
from torch.utils.tensorboard import SummaryWriter

import argparse

### hyperparameter
def get_args():
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument
    
    # E.g., --exp_id == '2021-10-01/01-17-47_debug'
    parser.add('--hyper_id', type=str, required=True)
    
    parser.add('--gpu_id', type=int, default=0)
    parser.add('--num_epochs', type=int, default = 100)
    parser.add('--batch_size', type=int, default = 512)
    parser.add('--num_workers', type=int, default = 4)
    parser.add('--learning_rate', type=int, default = 1e-4)
    parser.add('--loss_vertex_weight', type=int, default = 100)
    parser.add('--loss_landmark_weight', type=int, default=1)
    parser.add('--eye_loss_weight', type=int, default=1)
    parser.add('--mouth_loss_weight', type=int, default=1)

    parser.add('--exp_cfg', type=str, default = './config/config_EDTN.yaml')
    parser.add('--decapath', type=str, default = '../../DECA')
    parser.add('--coeff_path', type=str, default = './pretrained/transform_matrix_08_30.npy')

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

def read_mat(mat_file_path):
    """
    Mat file keys:
      - dict_keys(['id', 'exp', 'tex', 'angle', 'gamma', 'trans'])
    """
    mat_file = loadmat(mat_file_path)        
    pose = np.concatenate((mat_file['exp'], mat_file['angle'], mat_file['trans']), axis=-1).squeeze()
    # trans_params = np.array([float(item) for item in np.hsplit(mat_file['trans_params'], 5)])
    # pose = np.concatenate((pose, trans_params[2:]), axis=-1)
    return pose

def convert_angles_to_3dmm(rpy):
    radian_3dmm = (math.pi / 180) * rpy
    radian_3dmm = radian_3dmm[:, [1, 2, 0]]
    return radian_3dmm

def convert_poses_to_3dmm(pose):
    exp_pose = pose[:, :17]
    
    exp_coeffs = full_coeffs
    exp_coeffs = exp_coeffs.float().to(device)
  
    exp_3dmm = torch.mm(exp_pose, exp_coeffs)

    # 17 -> 50
    return exp_3dmm

def dict_npy2tensor(numpy_dict):
    tensor_dict = {}
    for k, v in numpy_dict.items():
        if isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        else:
            tensor_dict[k] = v
    return tensor_dict

def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, device, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        weight = torch.ones(real_2d_kp.size(0),real_2d_kp.size(1),1).to(device)
        real_2d_kp = torch.cat((real_2d_kp, weight), dim=-1)
        real_2d_kp[:,:,2] = weights[:]*real_2d_kp[:,:,2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

def dict_npy2tensor(numpy_dict):
    tensor_dict = {}
    for k, v in numpy_dict.items():
        if isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        else:
            tensor_dict[k] = v
    return tensor_dict

def custom_weighted_landmark_loss(predicted_landmarks, landmarks_gt, device, weight=1.):
    #smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    
    real_2d = landmarks_gt
    weights = torch.ones((68,)).cuda()
    # eye
    weights[36:48] = 5
    
    # inner mouth
    weights[60:68] = 5
    weights[48:60] = 5
    # side mouth
    weights[48] = 5
    weights[54] = 5

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, device, weights)
    return loss_lmk_2d * weight

# python train_EDTN.py --hyper_id=test_1
if __name__ == '__main__':
    args = get_args()

    full_coeffs = torch.from_numpy(np.load(f'{args.coeff_path}'))

    device = args.device
    # exp_id = '03-38-47_pirenderer' # AnimeCeleb Only
    # exp_cfg = f'../outputs/{exp_id}/.hydra/config_deca.yaml'
    cfg = load_config_file(args.exp_cfg, return_edict=True)
    cfg.train_params.batch_size = args.batch_size
    cfg.train_params.num_workers = args.num_workers
    cfg.dataset_folder = 'rotation'

    data_module = get_dm(cfg)
    data_module.setup()
    train_loader, valid_loader = data_module.train_dataloader(), data_module.valid_dataloader()

    log_dir = "./logs/" + args.hyper_id #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter(log_dir)

    deca2ani = Generator2ani()

    decar2ani = deca2ani.to(device)

    deca2ani_optimizer = torch.optim.Adam(params=deca2ani.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    # set to training mode
    deca2ani.train()

    ani_loss_avg = []
    val_ani_avg = []

    gen_vertex_loss_avg=[]
    gen_landmark_loss_avg=[]
    gen_eye_loss_avg=[]
    gen_mouth_loss_avg=[]

    sys.path.append(args.decapath)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from decalib.deca import DECA
    from decalib.datasets import datasets 
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from decalib.utils.tensor_cropper import transform_points

    # run DECA
    useTex = False
    rasterizer_type = 'standard'
    extractTex = False
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config = deca_cfg, device=device)

    if not os.path.exists(f'./logs/{args.hyper_id}/images'):
        os.mkdir(f'./logs/{args.hyper_id}/images')
        
    if not os.path.exists(f'./logs/{args.hyper_id}/weights'):
        os.mkdir(f'./logs/{args.hyper_id}/weights')

    print('Training...')

    for epoch in tqdm(range(args.num_epochs)):
        ani_loss_avg.append(0)
        val_ani_avg.append(0)
        
        gen_vertex_loss_avg.append(0)
        gen_landmark_loss_avg.append(0)
        gen_eye_loss_avg.append(0)
        gen_mouth_loss_avg.append(0)
        
        
        num_batches = 0
        num_valid_batches = 0

        for batch_idx, batch in enumerate(train_loader):

            batch['anime']['source'] = batch['anime']['source'][:, :3].to(device)
            batch['anime']['driving'] = torch.flip(batch['anime']['driving'][:, :3], dims=[0]).to(device)
            batch['anime']['driving_pose'] = torch.flip(batch['anime']['driving_pose'], dims=[0]).to(device)
            
            batch['vox']['source'] = batch['vox']['source'][:, :3].to(device)
            batch['vox']['driving'] = torch.flip(batch['vox']['driving'][:, :3], dims=[0]).to(device)
            batch['vox']['driving_pose'] = torch.flip(batch['vox']['driving_pose'], dims=[0]).to(device)
            

            ani_pose = batch['anime']['source_pose'][:,:17].to(device)

            ### 53dim
            dmm_pose = torch.cat((batch['vox']['driving_pose'][:,100:150], batch['vox']['driving_pose'][:,153:156]),dim=-1).to(device)


            ## discriminator loss

            set_requires_grad(deca2ani, requires_grad=True)

            deca2ani.train()

            deca2ani_optimizer.zero_grad()

            # gen_loss

            fake_pose = deca2ani(dmm_pose).squeeze()

            fake_dmm = convert_poses_to_3dmm(fake_pose)

            real_257 = torch.zeros(batch['vox']['driving_pose'].size(0),156)
            fake_257 = torch.zeros(batch['vox']['driving_pose'].size(0),156)

            real_257= real_257.to(device)
            fake_257= fake_257.to(device)

            real_257[:,100:150]=dmm_pose[:,:50].squeeze()
            real_257[:,153:156]=dmm_pose[:,50:53].squeeze()
            fake_257[:,100:150]=fake_dmm.squeeze()
            
            real_vertex = deca.vertex(real_257)

            fake_vertex = deca.vertex(fake_257)

            real_landmark=deca.landmark2d(real_257)

            fake_landmark=deca.landmark2d(fake_257)

            loss_vertex=F.mse_loss(real_vertex, fake_vertex)

            loss_landmark = custom_weighted_landmark_loss(fake_landmark,real_landmark,device)  # loss 재설정 하기 landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
            
            loss_eye=eyed_loss(fake_landmark,real_landmark)
        
            loss_mouth=lipd_loss(fake_landmark,real_landmark)
            
            loss_G= args.loss_vertex_weight*loss_vertex + args.loss_landmark_weight*loss_landmark + args.eye_loss_weight*loss_eye + args.mouth_loss_weight*loss_mouth
            
            loss_G.backward()

            deca2ani_optimizer.step()

            ani_loss_avg[-1] += loss_G.item()
            
            gen_vertex_loss_avg[-1] += loss_vertex.item()
            gen_landmark_loss_avg[-1] += loss_landmark.item()
            gen_eye_loss_avg[-1] += loss_eye.item()
            gen_mouth_loss_avg[-1] += loss_mouth.item()
            
            
            num_batches += 1

            global_step = epoch * len(train_loader) + batch_idx

            summary_writer.add_scalar('loss_G', loss_G, global_step)
            summary_writer.add_scalar('vertex', loss_vertex, global_step)
            summary_writer.add_scalar('landmark', loss_landmark, global_step)
            summary_writer.add_scalar('eye', loss_eye, global_step)
            summary_writer.add_scalar('mouth', loss_mouth, global_step)


        for batch_idx, batch in enumerate(valid_loader):
            batch['anime']['source'] = batch['anime']['source'][:, :3].to(device)
            batch['anime']['driving'] = torch.flip(batch['anime']['driving'][:, :3], dims=[0]).to(device)
            batch['anime']['driving_pose'] = torch.flip(batch['anime']['driving_pose'], dims=[0]).to(device)
            
            batch['vox']['source'] = batch['vox']['source'][:, :3].to(device)
            batch['vox']['driving'] = torch.flip(batch['vox']['driving'][:, :3], dims=[0]).to(device)
            batch['vox']['driving_pose'] = torch.flip(batch['vox']['driving_pose'], dims=[0]).to(device)
            
            ani_pose = batch['anime']['source_pose'][:,:17].to(device)

            dmm_pose = torch.cat((batch['vox']['driving_pose'][:,100:150], batch['vox']['driving_pose'][:,153:156]),dim=-1).to(device)


            set_requires_grad(deca2ani, requires_grad=False)

            deca2ani.eval()

            deca2ani_optimizer.zero_grad()

            # pose to dmm train

            # pose -> dmm with mlp
            fake_pose = deca2ani(dmm_pose).squeeze()

            fake_dmm = convert_poses_to_3dmm(fake_pose)

            real_257 = torch.zeros(batch['vox']['driving_pose'].size(0),156)
            fake_257 = torch.zeros(batch['vox']['driving_pose'].size(0),156)

            real_257= real_257.to(device)
            fake_257= fake_257.to(device)

            real_257[:,100:150]=dmm_pose[:,:50].squeeze()
            real_257[:,153:156]=dmm_pose[:,50:53].squeeze()
            fake_257[:,100:150]=fake_dmm.squeeze()
            
            real_vertex = deca.vertex(real_257)

            fake_vertex = deca.vertex(fake_257)

            real_landmark=deca.landmark2d(real_257)
            
            fake_landmark=deca.landmark2d(fake_257)

            loss_vertex=F.mse_loss(real_vertex, fake_vertex)

            loss_landmark = custom_weighted_landmark_loss(fake_landmark,real_landmark,device)  # loss 재설정 하기 landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
            
            loss_eye=eyed_loss(fake_landmark,real_landmark)
        
            loss_mouth=lipd_loss(fake_landmark,real_landmark)
        
            
            loss_G= args.loss_vertex_weight*loss_vertex + args.loss_landmark_weight*loss_landmark + args.eye_loss_weight*loss_eye + args.mouth_loss_weight*loss_mouth
            

            val_ani_avg[-1] += loss_G.item()

            num_valid_batches += 1

            global_step = epoch * len(valid_loader) + batch_idx

            summary_writer.add_scalar('valid_loss_G', loss_G, global_step)
            summary_writer.add_scalar('valid_vertex', loss_vertex, global_step)
            summary_writer.add_scalar('valid_landmark', loss_landmark, global_step)
            summary_writer.add_scalar('valid_eye', loss_eye, global_step)
            summary_writer.add_scalar('valid_mouth', loss_mouth, global_step)

            if epoch % 10 == 0:
                # visualization (make image)

                result = []

                natural_coeffs = torch.zeros(batch['vox']['driving_pose'].size(0),156).to(device)
                # source_pose = source_coeffs.clone()
                driving_coeffs = natural_coeffs.clone()
                converted_coeffs = natural_coeffs.clone()

                driving_pose = deca2ani(torch.cat((batch['vox']['driving_pose'][:,100:150], batch['vox']['driving_pose'][:,153:156]),dim=-1).to(device)).squeeze()
                
                converted_pose = convert_poses_to_3dmm(driving_pose).squeeze()

                driving_coeffs[:,100:150] = batch['vox']['driving_pose'][:,100:150]
                driving_coeffs[:,153:156] = batch['vox']['driving_pose'][:,153:156]

                converted_coeffs[:,100:150]= converted_pose

                torch.set_printoptions(precision=3, sci_mode=False)

                driving_rendered = deca.renderers(driving_coeffs) #, batch['vox']['driving'])

                converted_rendered = deca.renderers(converted_coeffs) #, batch['vox']['driving'])

                driving_rendered = driving_rendered.to('cpu')
                driving_rendered = driving_rendered.detach().numpy()

                converted_rendered = converted_rendered.to('cpu')
                converted_rendered = converted_rendered.detach().numpy()

                for i in range(batch['vox']['driving_pose'].size(0)):
                    

                    fig = plt.figure(2)
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax3 = fig.add_subplot(1, 2, 2)
                    # ax4 = fig.add_subplot(1, 4, 2)
                    # ax5 = fig.add_subplot(1, 4, 3)
                    # ax6 = fig.add_subplot(1, 4, 4)

                    ax1.axes.xaxis.set_visible(False)
                    ax1.axes.yaxis.set_visible(False)
                    ax3.axes.xaxis.set_visible(False)
                    ax3.axes.yaxis.set_visible(False)

                    ax1.set_title('driving_pose')
                    ax1.imshow(numpy2image(driving_rendered[i]))
                    ax3.set_title('transm_out')
                    ax3.imshow(numpy2image(converted_rendered[i]))

                    plt.show()
                    plt.savefig(f'./logs/{args.hyper_id}/images/example_{epoch}_{i}.png')
                    plt.close()

                    # if i == 10:
                    #     break


        if epoch % 10 == 0:
            torch.save(deca2ani.state_dict(), f'./logs/{args.hyper_id}/weights/{args.hyper_id}_{epoch}.pt')


        summary_writer.flush()
        ani_loss_avg[-1] /= num_batches
        val_ani_avg[-1] /= num_valid_batches
        
        gen_vertex_loss_avg[-1] /= num_batches
        gen_landmark_loss_avg[-1] /= num_batches
        gen_eye_loss_avg[-1] /= num_batches
        gen_mouth_loss_avg[-1] /= num_batches
        
        print('Epoch [%d / %d] average loss generator : %f , average val loss : %f' %
            (epoch+1, args.num_epochs, ani_loss_avg[-1], val_ani_avg[-1]))#, disc_loss_avg[-1]))
        print('Epoch [%d / %d] average vertex_loss generator : %f ' % (epoch+1, args.num_epochs, gen_vertex_loss_avg[-1]))
        print('Epoch [%d / %d] average landmark_loss generator : %f ' % (epoch+1, args.num_epochs, gen_landmark_loss_avg[-1]))
        print('Epoch [%d / %d] average eye_loss generator : %f ' % (epoch+1, args.num_epochs, gen_eye_loss_avg[-1]))
        print('Epoch [%d / %d] average mouth_loss generator : %f ' % (epoch+1, args.num_epochs, gen_mouth_loss_avg[-1]))
        
    print('dmm to pose train done.')   

    summary_writer.close()

    torch.save(deca2ani.state_dict(), f'./logs/{args.hyper_id}/weights/{args.hyper_id}.pt')