from .wrapper import DatasetModule

def get_dm(cfg):
    dm = DatasetModule(cfg)
    return dm