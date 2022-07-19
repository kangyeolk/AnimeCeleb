""" Ready for runner """
import torch

import os

from runners.fomm import RunnerFOMM
from runners.pirender import RunnerPIRender
from runners.animo import RunnerAnimo

from models import get_model_pack
from utils.functions import load_ckpt


def get_runner(cfg):
    runner = None

    ##### Ready for running
    # Set model-pack
    start_epoch = 0 
    model_pack = get_model_pack(cfg)

    # Set checkpoint
    if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
        ckpt_data = torch.load(cfg.resume_path)
        model_pack = load_ckpt(cfg, model_pack, ckpt_data)
        print('Load pre-trained checkpoint is done...!')
        
    if cfg.method == 'fomm':
        runner = RunnerFOMM(cfg, model_pack, start_epoch)
    elif cfg.method == 'pirender':
        runner = RunnerPIRender(cfg, model_pack, start_epoch)
    elif cfg.method == 'animo':
        runner = RunnerAnimo(cfg, model_pack, start_epoch)
    
    return runner

def get_runner_class(cfg):
    _class = None
    if cfg.method == 'fomm':
        _class = RunnerFOMM
    elif cfg.method == 'pirender':
        _class = RunnerPIRender
    elif cfg.method == 'animo':
        _class = RunnerAnimo
    
    return _class
