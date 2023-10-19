from PIL import Image

import numpy as np
import torch
from torch.autograd import Function
from skimage import io, img_as_float32
import os
import math
from scipy.io import loadmat

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
def dict_npy2tensor(numpy_dict):
    tensor_dict = {}
    for k, v in numpy_dict.items():
        if isinstance(v, np.ndarray):
            tensor_dict[k] = torch.from_numpy(v).to(device)
        else:
            tensor_dict[k] = v
    return tensor_dict

def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
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

def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight

def eye_dis(landmarks):
    # left eye:  [38,42], [39,41] - 1
    # right eye: [44,48], [45,47] -1
    eye_up = landmarks[:,[37, 38, 43, 44], :]
    eye_bottom = landmarks[:,[41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom)**2).sum(2)) #[bz, 4]
    return dis

def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[:,:,:2])
    gt_eyed = eye_dis(real_2d[:,:,:2])

    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss

def lip_dis(landmarks):
    # up inner lip:  [62, 63, 64] - 1
    # down innder lip: [68, 67, 66] -1

    ## inner lip
    # lip_up = landmarks[:,[61, 62, 63], :]
    # lip_down = landmarks[:,[67, 66, 65], :] 

    ## upper lip
    lip_up = landmarks[:,[50, 51, 52], :]
    lip_down = landmarks[:,[58, 57, 56], :]
    dis = torch.sqrt(((lip_up - lip_down)**2).sum(2)) #[bz, 4]
    return dis

def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:,:,:2])
    gt_lipd = lip_dis(real_2d[:,:,:2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss

def numpy2image(array):
    array = np.squeeze(array)
    array = array.transpose(1,2,0)
    return array

def custom_weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
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

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight

def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight

def convert_poses_to_3dmm(pose):
    exp_pose = pose[:, :17]
    
    exp_coeffs = full_coeffs
    exp_coeffs = exp_coeffs.float().to(device)
  
    exp_3dmm = torch.mm(exp_pose, exp_coeffs)

    # 17 -> 50
    return exp_3dmm

def l1_distance(gt,pred):
    div = frame_shape[0]*frame_shape[1]*frame_shape[2]
    dif_abs = torch.abs(gt - pred).sum(1).sum(1).sum(1)
    dif_abs = dif_abs/div
    return dif_abs

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')

def read_deca(mat_file_path):
    """
    Mat file keys:
      - dict_keys(['id', 'exp', 'tex', 'angle', 'gamma', 'trans'])
    """
    mat_file = loadmat(mat_file_path)        
    pose = np.concatenate((mat_file['shape'], mat_file['exp'], mat_file['pose']), axis=-1).squeeze()
    # trans_params = np.array([float(item) for item in np.hsplit(mat_file['trans_params'], 5)])
    # pose = np.concatenate((pose, trans_params[2:]), axis=-1)
    return pose

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

def convert_deca_to_angles(pose):
    # [DECA] roll, yaw, pitch (radian) --> [Pose vec] roll, pitch, yaw (angle)
    # degree_pose = pose.unsqueeze(0)
    # degree_pose = (180 / math.pi) * pose
    degree_pose = pose[:,[0, 2, 1]] * (180 / math.pi)
    return degree_pose

    
    # samples_per_identity = 16
    samples_per_identity = 10
    entire_pose = []
    entire_video= []

    # name = videos_vox[identity_vox_sample]
    # path_pose = np.random.choice(glob.glob(os.path.join(pose_dir_vox, name + '*.mp4')))

    path_video = path_pose.replace('3dmm', 'images')
    frames = sorted(os.listdir(str(path_pose)))
    num_frames = len(frames)
    frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=samples_per_identity))
    # frame_idx = np.arange(num_frames)
    video_array_vox = [img_as_float32(io.imread(os.path.join(path_video, frames[idx].replace('.mat', '.png')))) for idx in frame_idx]
    pose_array_vox = [read_mat(os.path.join(path_pose, frames[idx])) for idx in frame_idx]
    entire_video.append(np.array(video_array_vox))
    entire_pose.append(np.array(pose_array_vox))

    entire_video = np.concatenate(entire_video, axis=0)
    entire_pose = np.concatenate(entire_pose, axis=0)
    
    return entire_video, entire_pose

def one_vox_deca_pose(path_pose):
    
    samples_per_identity = 30

    entire_pose = []
    entire_video= []

    # name = videos_vox[identity_vox_sample]
    # path_pose = np.random.choice(glob.glob(os.path.join(pose_dir_vox, name + '*.mp4')))

    path_video = path_pose.replace('deca', 'images')
    # path_rotation = path_pose.replace('deca','3dmm')
    frames = sorted(os.listdir(str(path_pose)))
    num_frames = len(frames)
    # frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=samples_per_identity))
    frame_idx = np.arange(num_frames)
    video_array_vox = [img_as_float32(io.imread(os.path.join(path_video, frames[idx].replace('.mat', '.png')))) for idx in frame_idx]
    pose_array_vox = [read_deca(os.path.join(path_pose, frames[idx])) for idx in frame_idx]
    # pose_array_rotation = [read_deca(os.path.join(path_pose, frames[idx])) for idx in frame_idx]
    np_pose_array_vox = np.array(pose_array_vox)
    # np_pose_array_rotation = np.array(pose_array_rotation)
    # np_pose_array_vox[:,153:156] = np_pose_array_rotation[:,64:67]

    entire_video.append(np.array(video_array_vox))
    entire_pose.append(np_pose_array_vox)


    entire_video = np.concatenate(entire_video, axis=0)
    entire_pose = np.concatenate(entire_pose, axis=0)
    
    return entire_video, entire_pose

def get_anime_basis(anime_image_path):
    entire_video = []
    frame_shape=(256, 256, 3)
    video_array_anime = [io.imread(anime_image_path)]#  video_items[idx])) for idx in frame_idx] # [0-1]
    video_array_anime = [img_as_float32(np.resize(rgba2rgb(v), frame_shape)) for v in video_array_anime]
    entire_video.append(np.array(video_array_anime)) # .transpose((2, 0, 1)))
    
    entire_video = np.concatenate(entire_video, axis=0)

    return entire_video