import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()

    def forward(self, input):
        return input