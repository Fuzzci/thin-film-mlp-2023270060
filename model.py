import torch
import torch.nn as nn
from config import MLP_HIDDEN


class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=41, hidden=MLP_HIDDEN):
        super().__init__()

        layers = []
        prev = input_dim

        # 依次添加隐藏层：128 -> 128 -> 64
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        # 输出层
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)