import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(SAMLayer, self).__init__()

        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        # spatial attention
        self.sifmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=0, keepdim=True)
        avg_out = torch.mean(x, dim=0, keepdim=True)
        Fs = torch.cat([max_out, avg_out], dim=0)
        Fs = Fs.permute(1,0,2)
        Fs = self.conv(Fs)
        spatial_out = self.sifmoid(Fs)
        spatial_out = spatial_out.permute(1,0,2)
        x = spatial_out * x
        return x
