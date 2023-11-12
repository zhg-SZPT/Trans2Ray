import torch
from torch import nn


class MAWS(nn.Module):
    # mutual attention weight selection
    def __init__(self):
        super(MAWS, self).__init__()

    def forward(self, x, contributions):
        length = x.size()[1]

        contributions = contributions.mean(1)
        weights = x[:,:,0,:].mean(1)

        scores = contributions*weights

        max_inx = torch.argsort(scores, dim=1,descending=True)


        return None, max_inx