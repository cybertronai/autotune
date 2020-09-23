import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mlp_autoencoder_deep']


class MLP_AUTOENCODER_DEEP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        n_hid = 1024
        n_bottleneck = 196
        self.l1 = nn.Linear(28*28, n_hid, bias=False)
        self.l2 = nn.Linear(n_hid, n_hid, bias=False)
        self.l3 = nn.Linear(n_hid, n_hid, bias=False)
        self.l4 = nn.Linear(n_hid, n_bottleneck, bias=False)
        self.l5 = nn.Linear(n_bottleneck, n_hid, bias=False)
        self.l6 = nn.Linear(n_hid, n_hid, bias=False)
        self.l7 = nn.Linear(n_hid, n_hid, bias=False)
        self.l8 = nn.Linear(n_hid, 28*28, bias=False)

    def forward(self, x: torch.Tensor):
        x = x.view([-1, 28*28])
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        return x


def mlp_autoencoder_deep(**kwargs):
    model = MLP_AUTOENCODER_DEEP(**kwargs)
    return model

