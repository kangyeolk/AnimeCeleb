import torch
import torch.nn as nn

class GANLoss(nn.Module):

    def __init__(self, weight):
        self.weight = weight

    def forward(self, x: torch.Tensor):
        return x 


