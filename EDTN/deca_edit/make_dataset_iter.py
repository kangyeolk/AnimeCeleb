
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder_first = args.savefolder
    device = args.device
    os.makedirs(savefolder_first, exist_ok=True)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)


    pose_videos = sorted(os.listdir(args.inputpath))
    videos_vox = {os.path.basename(video) for video in pose_videos}
    videos_vox = sorted(list(videos_vox))


    # load test images 

    idx = 0
    # for i in range(len(testdata)):
    for video_name in tqdm(videos_vox):
        
        idx += 1
        testdata_path = os.path.join(args.inputpath, video_name)
        testdata = datasets.TestData(testdata_path, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

        savefolder_second = os.path.join(savefolder_first, video_name)
        os.makedirs(savefolder_second, exist_ok=True)

        for i in range(len(testdata)):
            name = testdata[i]['imagename']
            images = testdata[i]['image'].to(device)[None,...]
            with torch.no_grad():
                codedict = deca.encode(images, use_detail=True)
                codedict = util.dict_tensor2npy(codedict)
                savemat(os.path.join(savefolder_second, name+'.mat'), codedict)



    print(f'-- please check the results in {savefolder_first}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='/home/nas2_userF/dataset/anime_talk/video-preprocessing/vox/images/test', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='/home/nas2_userF/dataset/anime_talk/video-preprocessing/vox/deca_detail/test', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # save
    parser.add_argument('--saveExp', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save encoder outputs as .mat' )
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    main(parser.parse_args())