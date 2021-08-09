import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):

        return x

if __name__ == '__main__':
    net = Model()
    x = torch.randn((64, 3, 512, 512))
    y_pred = net(x)
    print(y_pred.shape)
