import torch
import torch.nn as nn
import torch.nn.functional as F

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=True)
        self.in1 = nn.InstanceNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.in3 = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 1, 1, bias=False)
        self.in4 = nn.InstanceNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = F.leaky_relu(self.in1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.in2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.in3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.in4(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x

class Residual(nn.Module):
    def __init__(self, dim, use_bias):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual

class Gen(nn.Module):
    def __init__(self, base=64, res_num=7):
        super().__init__()
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base, 7, 1, 0, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.down = nn.Sequential(
            nn.Conv2d(base, base*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )

        self.res = nn.Sequential(*[Residual(base*4, False) for _ in range(res_num)])

        # ========== 这里修复了！==========
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, 3, 7, 1, 0),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.head(x)
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        x = self.tail(x)
        return x
