from torch import nn
import torch.nn.functional as F


class Get_feature(nn.Module):
    def __init__(self,input_dim):
        super(Get_feature, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        x = x.permute(1,2,0)
        x = F.adaptive_avg_pool1d(x, 1).permute(0, 2, 1)
        b_f = x
        x = x.flatten(1)
        x = self.classifier(x)
        return x, b_f
