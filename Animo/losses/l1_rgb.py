import torch.nn as nn

class L1RGB(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, batch, output):
        
        real_rgb = batch['target']
        fake_rgb = output['pred']

        return self.weight * nn.functional.l1_loss(fake_rgb, real_rgb[:, 0])