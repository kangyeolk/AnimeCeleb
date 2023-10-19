import pytorch_lightning as pl
from .augmentation import *
from albumentations import *
from torch.utils.data import DataLoader

from data.animeceleb import AnimeCelebDataset, AnimeCelebAndVoxDataset, DatasetRepeater, AnimeCelebAndDecaDataset


class DatasetModule(pl.LightningDataModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset_name

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        if stage == 'fit' or stage is None:
            if self.dataset_name == 'animeceleb':
                self.train_set = AnimeCelebDataset(mode='train', dataset_folder=self.cfg.dataset_folder, **self.cfg['dataset_params'])
                self.train_set = self.repeat_dataset(self.train_set)
                self.valid_set = AnimeCelebDataset(mode='valid', **self.cfg['dataset_params'])
            elif self.dataset_name == 'animeceleb_vox':
                self.train_set = AnimeCelebAndVoxDataset(mode='train', **self.cfg['dataset_params'])
                self.train_set = self.repeat_dataset(self.train_set)
                self.valid_set = AnimeCelebAndVoxDataset(mode='valid', **self.cfg['dataset_params'])
            elif self.dataset_name == 'animeceleb_deca':
                self.train_set = AnimeCelebAndDecaDataset(mode='train', **self.cfg['dataset_params'])
                self.train_set = self.repeat_dataset(self.train_set)
                self.valid_set = AnimeCelebAndDecaDataset(mode='valid', **self.cfg['dataset_params'])

    def repeat_dataset(self, dataset):
        return DatasetRepeater(dataset, self.cfg.train_params.num_repeats)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          num_workers=self.cfg.train_params.num_workers,
                          batch_size=self.cfg.train_params.batch_size,
                          shuffle=True, 
                          drop_last=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_set,
                          num_workers=self.cfg.train_params.num_workers,
                          batch_size=self.cfg.train_params.batch_size,
                          shuffle=False,
                          drop_last=False)