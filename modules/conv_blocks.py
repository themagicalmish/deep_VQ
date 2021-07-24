import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.sc = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        x = self.down(x)
        sc = self.sc(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = (x + sc) * (1/np.sqrt(2))

        return x

class DownBlock3D(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.down = nn.MaxPool3d(2)
        self.conv1 = nn.Conv3d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.sc = nn.Conv3d(c_in, c_out, 1)

    def forward(self, x):
        x = self.down(x)
        sc = self.sc(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = (x + sc) * (1/np.sqrt(2))

        return x

class UpBlock3D(nn.Module):
    def __init__(self, c_in, c_out, two_convs=True):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, 1)

        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.two_convs = two_convs
        if two_convs:
            self.conv3 = nn.Conv3d(c_out, c_out, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        sc = x
        x = self.conv2(x)
        x = F.leaky_relu(x)
        if self.two_convs:
            x = self.conv3(x)
            x = F.leaky_relu(x)
            x = (x + sc) * (np.sqrt(2))
        return x

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, two_convs=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.two_convs = two_convs
        if two_convs:
            self.conv3 = nn.Conv2d(c_out, c_out, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        sc = x
        x = self.conv2(x)
        x = F.leaky_relu(x)
        if self.two_convs:
            x = self.conv3(x)
            x = F.leaky_relu(x)
            x = (x + sc) * (np.sqrt(2))
        return x

class UpBlockShortCut(nn.Module):
    def __init__(self, c_in, c_out, two_convs=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1)

        self.up = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.conv3 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.two_convs = two_convs

    def forward(self, x):
        x, s = x
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        sc = x
        x = (x + s) * (1/np.sqrt(2))
        x = self.conv2(x)
        x = F.leaky_relu(x)
        if self.two_convs:
            x = self.conv3(x)
            x = F.leaky_relu(x)
            x = (x + sc) * (np.sqrt(2))
        return x


class ModConv2d(nn.Module):
    def __init__(self, c_in, c_out, style_d, demod=True):
        super().__init__()
        f = 3
        self.c_in = c_in
        self.c_out = c_out
        self.f = f
        self.demod = demod
        self.w = nn.Parameter(torch.empty(
            c_out, c_in, f, f), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(c_out, 1, 1), requires_grad=True)
        self.style_d = style_d
        nn.init.xavier_uniform_(self.w)
        self.affine = nn.Linear(style_d, self.c_in)

    def forward(self, x):
        x, s = x
        bs = x.shape[0]
        s = self.affine(s)
        s = s.reshape([s.shape[0], 1, self.c_in, 1, 1])

        wo = self.w[None, ...]
        w = wo * (s + 1)

        if self.demod:
            d = (w.square().sum(dim=[2, 3, 4], keepdim=True) + 1e-8).sqrt()
            w = w / d

        x = x.reshape([1, -1, x.shape[2], x.shape[3]])
        conv_w = w.reshape(
            [-1, w.shape[2], w.shape[3], w.shape[4]]
        )
        x = F.conv2d(x, conv_w, stride=1, groups=bs, padding=1)
        x = x.reshape([bs, self.c_out, x.shape[2], x.shape[3]])
        x += self.b
        return x


class ModConv3d(nn.Module):
    def __init__(self, c_in, c_out, style_d, demod=True):
        super().__init__()
        f = 3
        self.c_in = c_in
        self.c_out = c_out
        self.f = f
        self.demod = demod
        self.w = nn.Parameter(torch.empty(
            c_out, c_in, f, f, f), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(c_out, 1, 1, 1), requires_grad=True)
        self.style_d = style_d
        nn.init.xavier_uniform_(self.w)
        self.affine = nn.Linear(style_d, self.c_in)

    def forward(self, x):
        x, s = x
        bs = x.shape[0]
        s = self.affine(s)
        s = s.reshape([s.shape[0], 1, self.c_in, 1, 1, 1])

        wo = self.w[None, ...]
        w = wo * (s + 1)

        if self.demod:
            d = (w.square().sum(dim=[2, 3, 4, 5], keepdim=True) + 1e-8).sqrt()
            w = w / d

        x = x.reshape([1, -1, x.shape[2], x.shape[3], x.shape[4]])
        conv_w = w.reshape(
            [-1, w.shape[2], w.shape[3], w.shape[4], w.shape[5]]
        )

        x = F.conv3d(x, conv_w, stride=1, groups=bs, padding=1)
        x = x.reshape([bs, self.c_out, x.shape[2], x.shape[3], x.shape[4]])
        x += self.b
        return x