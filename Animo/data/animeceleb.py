import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage import io, img_as_float32

from torch.utils.data import Dataset

from data.augmentation import AllAugmentationTransform



def read_mat(mat_file_path):
    """
    Mat file keys:
      - dict_keys(['id', 'exp', 'tex', 'angle', 'gamma', 'trans'])
    """
    mat_file = loadmat(mat_file_path)        
    pose = np.concatenate((mat_file['exp'], mat_file['angle'], mat_file['trans']), axis=-1).squeeze()

    return pose


class AnimeCelebDataset(Dataset):

    def __init__(self, root_dir, frame_shape=(256, 256, 3), 
                 id_sampling=False, mode='train', dataset_folder='rotation',
                 return_basis=False, random_seed=0, augmentation_params=None):
        self.root_dir = root_dir
        
        self.file_csv = pd.read_csv(os.path.join(root_dir, 'combined_pose.csv'))        
        print(f"Reading dataset information done! There are {len(self.file_csv)} files")
        
        self.image_path = os.path.join(root_dir, dataset_folder)     
        self.frame_shape = tuple(frame_shape)
                
        self.return_basis = return_basis
        self.basis_path = os.path.join(root_dir, 'basis')
        
        self.id_sampling = id_sampling
        self.is_train = True if mode == 'train' else False
        
        json_file = os.path.join(root_dir, 'cached.json')
        if not os.path.exists(json_file):
            print('There is no cached.json saving it for later, it will be used later...!')
            self.videos_tree = self.gather_same_identity_images()
            with open(json_file,'w') as f:
                json.dump(self.videos_tree, f)
        else:
            print('There is a cached.json, use it here...!')
            with open(json_file) as f:
                self.videos_tree = json.load(f)            
        
        self.id_keys = list(self.videos_tree.keys())
        
        # Read Train & Test list   
        if os.path.isfile('../../../data/train_list.txt'):
            self.train_ids = [line[:-1] for line in open('../../../data/train_list.txt', 'r')]
            self.test_ids = [line[:-1] for line in open('../../../data/test_list.txt', 'r')]
        elif os.path.isfile('../data/train_list.txt'):        
            self.train_ids = [line[:-1] for line in open('../data/train_list.txt', 'r')]
            self.test_ids = [line[:-1] for line in open('../data/test_list.txt', 'r')]
        elif os.path.isfile('./data/train_list.txt'):        
            self.train_ids = [line[:-1] for line in open('./data/train_list.txt', 'r')]
            self.test_ids = [line[:-1] for line in open('./data/test_list.txt', 'r')]

        print(f"Reading dataset done! There are {len(self.train_ids)} identities to train and {len(self.test_ids)} identities to test!")
        
        if mode == 'train':
            self.using_ids = self.train_ids
        elif mode == 'valid':
            self.using_ids = self.test_ids
        
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None    
    
    def gather_same_identity_images(self):
        self.identity_numbers = np.unique(list(self.file_csv['org_id'])).tolist()
        videos = {}
        for idx, number in enumerate(self.identity_numbers):
            videos[number] = {'num_cluster': None, 'folder_name': None, 'img_names': [], 'pose_information': []}
        for line_num in range(len(self.file_csv)):
            line = list(self.file_csv.iloc[line_num])
            if videos[line[0]]['num_cluster'] is None:
                videos[line[0]]['num_cluster'] = str(line[1])                # num_cluster            
            if videos[line[0]]['folder_name'] is None:
                videos[line[0]]['folder_name'] = line[2].split('_')[0]  # png_name -> folder_name
            videos[line[0]]['img_names'].append(line[2])                # png_name
            videos[line[0]]['pose_information'].append(line[3:])
        return videos

    def __len__(self):
        return len(self.using_ids)

    
    def __getitem__(self, idx):
        
        identity_to_sample = self.using_ids[idx]
        video_name = identity_to_sample
        num_cluster = self.videos_tree[identity_to_sample]['num_cluster']
        folder_name = self.videos_tree[identity_to_sample]['folder_name']
        video_items = self.videos_tree[identity_to_sample]['img_names']
        pose_items = self.videos_tree[identity_to_sample]['pose_information']
                            
        # Get morphed images
        num_frames = len(video_items)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        video_array = [io.imread(os.path.join(self.image_path, num_cluster, folder_name, video_items[idx])) for idx in frame_idx] # [0-1]
        video_array = [img_as_float32(rgba2rgb(v)) for v in video_array]
        
        # Get poses
        pose_array = [pose_items[idx] for idx in frame_idx]
        
        # Get basis images
        if self.return_basis:
            basis_img = io.imread(os.path.join(self.basis_path, folder_name, 'basis.png'))
            basis_img = [img_as_float32(rgba2rgb(basis_img))]


        if self.transform is not None:
            video_array = self.transform(video_array)
            if self.return_basis:
                basis_img = self.transform(basis_img)
        
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            source_pose = np.array(pose_array[0], dtype='float32')
            driving_pose = np.array(pose_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_pose'] = driving_pose
            out['source_pose'] = source_pose
            if self.return_basis:
                basis = np.array(basis_img[0], dtype='float32')
                out['basis'] = basis.transpose((2, 0, 1))

        else:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            source_pose = np.array(pose_array[0], dtype='float32')
            driving_pose = np.array(pose_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_pose'] = driving_pose
            out['source_pose'] = source_pose
            if self.return_basis:
                basis = np.array(basis_img[0], dtype='float32')
                out['basis'] = basis.transpose((2, 0, 1))

        out['name'] = video_name
        
        return out

    
class AnimeCelebAndVoxDataset(Dataset):

    def __init__(self, root_dir_anime, root_dir_vox, frame_shape=(256, 256, 3), 
                 id_sampling=False, mode='train', dataset_folder='rotation', augmentation_params=None):
        self.is_train = True if mode == 'train' else False
        self.root_dir_anime = root_dir_anime
        self.root_dir_vox = os.path.join(root_dir_vox, 'train' if self.is_train else 'test')
        self.pose_dir_vox = self.root_dir_vox.replace('images', '3dmm')
        self.basis_path = os.path.join(root_dir_anime, 'basis')
        
        # AnimeCeleb Pose Information
        self.file_csv = pd.read_csv(os.path.join(root_dir_anime, 'combined_pose.csv'))        
        print(f"Reading dataset information done! There are {len(self.file_csv)} files")
        
        self.image_path_anime = os.path.join(root_dir_anime, dataset_folder)
        self.frame_shape = tuple(frame_shape)
        
        self.id_sampling = id_sampling
        
        # Save Cache.json of AnimeCeleb
        json_file = os.path.join(root_dir_anime, 'cached.json')
        if not os.path.exists(json_file):
            print('There is no cached.json saving it for later, it will be used later...!')
            self.videos_tree_anime = self.gather_same_identity_images()
            with open(json_file,'w') as f:
                json.dumps(self.videos_tree_anime, f)
        else:
            print('There is a cached.json, use it here...!')
            with open(json_file) as f:
                self.videos_tree_anime = json.load(f)
        
        # AnimeCeleb Video List
        self.id_keys_anime = list(self.videos_tree_anime.keys())
        
        # Read Train & Test list   
        if os.path.isfile('../../../data/train_list.txt'):
            self.train_ids_anime = [line[:-1] for line in open('../../../data/train_list.txt', 'r')]
            self.test_ids_anime = [line[:-1] for line in open('../../../data/test_list.txt', 'r')]
        elif os.path.isfile('../data/train_list.txt'):        
            self.train_ids_anime = [line[:-1] for line in open('../data/train_list.txt', 'r')]
            self.test_ids_anime = [line[:-1] for line in open('../data/test_list.txt', 'r')]
        elif os.path.isfile('./data/train_list.txt'):        
            self.train_ids_anime = [line[:-1] for line in open('./data/train_list.txt', 'r')]
            self.test_ids_anime = [line[:-1] for line in open('./data/test_list.txt', 'r')]

        # self.train_ids_anime, self.test_ids_anime = train_test_split(self.id_keys_anime, random_state=random_seed, test_size=0.2)
        print(f"Reading dataset done! There are {len(self.train_ids_anime)} identities to train and {len(self.test_ids_anime)} identities to test!")
        
        if mode == 'train':
            self.using_ids_anime = self.train_ids_anime
        elif mode == 'valid':
            np.random.seed(0) # Fix frame sample for validation
            self.using_ids_anime = self.test_ids_anime
            
        # VoxCeleb Video List
        if id_sampling and (mode == 'train'):
            videos_vox = {os.path.basename(video).split('#')[0] for video in os.listdir(self.pose_dir_vox)}
            self.videos_vox = list(videos_vox)
        elif id_sampling and (mode == 'valid'):
            pose_videos = sorted(os.listdir(self.pose_dir_vox))
            videos_vox = {os.path.basename(video).split('#')[0] for video in pose_videos}
            self.videos_vox = sorted(list(videos_vox))
            # self.videos_vox = os.listdir(self.pose_dir_vox)
        
        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None    
    
    def gather_same_identity_images(self):
        self.identity_numbers = np.unique(list(self.file_csv['org_id'])).tolist()
        videos = {}
        for idx, number in enumerate(self.identity_numbers):
            videos[number] = {'num_cluster': None, 'folder_name': None, 'img_names': [], 'pose_information': []}
        for line_num in range(len(self.file_csv)):
            line = list(self.file_csv.iloc[line_num])
            if videos[line[0]]['num_cluster'] is None:
                videos[line[0]]['num_cluster'] = line[1]                # num_cluster            
            if videos[line[0]]['folder_name'] is None:
                videos[line[0]]['folder_name'] = line[2].split('/')[0]  # png_name -> folder_name
            videos[line[0]]['img_names'].append(line[2])                # png_name
            videos[line[0]]['pose_information'].append(line[3:])
        return videos

    def __len__(self):
        return min(len(self.using_ids_anime), len(self.videos_vox))
    
    def __getitem__(self, idx):
        
        out = {'anime': {}, 'vox': {}}
        
        # Load VoxCeleb Frames
        if self.is_train and self.id_sampling:
            name = self.videos_vox[idx]
            path_pose = np.random.choice(glob.glob(os.path.join(self.pose_dir_vox, name + '*.mp4')))
            path_video = path_pose.replace('3dmm', 'images')
        elif not self.is_train and self.id_sampling:            
            name = self.videos_vox[idx]
            path_pose = np.random.choice(glob.glob(os.path.join(self.pose_dir_vox, name + '*.mp4')))
            path_video = path_pose.replace('3dmm', 'images')
        
        video_name = os.path.basename(path_video)
        out['vox']['name'] = video_name
        
        frames = sorted(os.listdir(str(path_pose)))
        num_frames = len(frames)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        video_array_vox = [img_as_float32(io.imread(os.path.join(path_video, frames[idx].replace('.mat', '.png')))) for idx in frame_idx]
        pose_array_vox = [read_mat(os.path.join(path_pose, frames[idx])) for idx in frame_idx]
                
        # Load AnimeCeleb Frames
        anime_idx = np.random.randint(0, len(self.using_ids_anime) - 1, 1)[0]
        identity_to_sample = self.using_ids_anime[anime_idx]
        video_name = identity_to_sample
        num_cluster = self.videos_tree_anime[identity_to_sample]['num_cluster']
        folder_name = self.videos_tree_anime[identity_to_sample]['folder_name']
        video_items = self.videos_tree_anime[identity_to_sample]['img_names']
        pose_items = self.videos_tree_anime[identity_to_sample]['pose_information']
        out['anime']['num_cluster'] = num_cluster
        out['anime']['folder_name'] = folder_name
        
        num_frames = len(video_items)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
        video_array_anime = [io.imread(os.path.join(self.image_path_anime, num_cluster, folder_name, video_items[idx])) for idx in frame_idx] # [0-1]
        video_array_anime = [img_as_float32(rgba2rgb(v)) for v in video_array_anime]
        pose_array_anime = [pose_items[idx] for idx in frame_idx] # [0-1]
        
        # Data Transformation & Augmentation
        if self.transform is not None:
            video_array_anime = self.transform(video_array_anime)
            video_array_vox = self.transform(video_array_vox)
        
        # Data of AnimeCeleb
        source_anime = np.array(video_array_anime[0], dtype='float32')
        driving_anime = np.array(video_array_anime[1], dtype='float32')
        source_pose_anime = np.array(pose_array_anime[0], dtype='float32')
        driving_pose_anime = np.array(pose_array_anime[1], dtype='float32')
        
        out['anime']['source'] = source_anime.transpose((2, 0, 1))
        out['anime']['driving'] = driving_anime.transpose((2, 0, 1))
        out['anime']['source_pose'] = source_pose_anime
        out['anime']['driving_pose'] = driving_pose_anime
        
        # Data of VoxCeleb
        source_vox = np.array(video_array_vox[0], dtype='float32')
        driving_vox = np.array(video_array_vox[1], dtype='float32')
        source_pose_vox = np.array(pose_array_vox[0], dtype='float32')
        driving_pose_vox = np.array(pose_array_vox[1], dtype='float32')
        
        out['vox']['source'] = source_vox.transpose((2, 0, 1))
        out['vox']['driving'] = driving_vox.transpose((2, 0, 1))
        out['vox']['source_pose'] = source_pose_vox
        out['vox']['driving_pose'] = driving_pose_vox
        
        return out
    

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


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
