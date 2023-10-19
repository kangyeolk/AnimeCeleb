# Built-in
import os
import logging
import sys
# sys.path.insert(0, '../Animo')
# Deep-Learning Framework
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

# Modules
from runners import get_runner
from data import get_dm
from utils.callbacks import get_checkpoint_callback

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
PATH = os.getcwd()


@hydra.main(config_path="./configs/", config_name="Anime_generator.yaml")
def interactive_run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    # Get data, model, logger
    data_module = get_dm(cfg)
    data_module.setup()
    runner = get_runner(cfg)
    logger = TensorBoardLogger('tb_logs')
    cfg.working_dir = PATH # reset working directory to root directory of proj.
    
    # Setup GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpu))
    cfg.gpu = list(range(len(cfg.gpu)))

    # Set trainer
    checkpoint_callback = get_checkpoint_callback(criterion='epoch', save_frequency=cfg.train_params.checkpoint_freq)
    
    if cfg.resume_path == 'None':
        trainer = pl.Trainer(gpus=cfg.gpu, 
                             deterministic=False, 
                             max_epochs=cfg.train_params.num_epochs, 
                             callbacks=[checkpoint_callback],
                             logger=logger,
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=2,
                             accelerator="ddp",
                             precision=32)
    else:
        trainer = pl.Trainer(gpus=cfg.gpu, 
                             deterministic=False, 
                             max_epochs=cfg.train_params.num_epochs, 
                             callbacks=[checkpoint_callback],
                             resume_from_checkpoint=cfg.resume_path, # Load Checkpoint
                             logger=logger,
                             check_val_every_n_epoch=1,
                             num_sanity_val_steps=2,
                             accelerator="ddp",
                             precision=32)
    
    print(runner)
    
    # Train
    trainer.fit(runner,
                data_module.train_dataloader(),
                data_module.valid_dataloader())

if __name__ == "__main__":
    interactive_run()