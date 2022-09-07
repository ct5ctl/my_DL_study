import imp
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)