from torch import nn


class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=256, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        #
        # self.global_att2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels),
        # )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.local_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + y * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.local_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = y * (1 - wei2)
        return xo