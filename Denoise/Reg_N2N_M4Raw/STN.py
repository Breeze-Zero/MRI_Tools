import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional, Tuple
class SpatialTransformer_old(nn.Module):
    def __init__(self, in_chans,spatial_dims):
        super(SpatialTransformer_old, self).__init__()
        self._h, self._w = spatial_dims 
        self.fc1 = nn.Linear(in_chans*self._h*self._w, 1024) # 可根据自己的网络参数具体设置
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x): 
        batch_images = x #保存一份原始数据
        B,C,H,W = x.shape
        x = x.view(-1, C*H*W)
        # 利用FC结构学习到6个参数
        x = self.fc1(x)
        x = self.fc2(x) 
        x = x.view(-1, 2,3) # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points,align_corners=False)
        return rois, affine_grid_points
    
class SpatialTransformer_shift(nn.Module):
    def __init__(self):
        super(SpatialTransformer_shift, self).__init__()
        self.fc1 = nn.Linear(1, 6) 

    def forward(self, x,shift): 
        B,C,H,W = x.shape
        shift = self.fc1(shift)
        shift = shift.view(-1, 2,3) # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(shift, torch.Size((B,C,H,W)), align_corners=False)
        # 将采样点作用到原始数据上
        rois = F.grid_sample(x, affine_grid_points,align_corners=False)
        return rois#, affine_grid_points
    
class SpatialTransformer(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.net = torch.nn.Sequential( \
                UNet(2*channels, 32, (32, 64, 64, 64, 64)), \
                torch.nn.LeakyReLU(inplace=True), \
                torch.nn.Conv2d(32, 2, kernel_size=3, padding=1))
        with torch.no_grad():
            for param in self.net.parameters():
                param = param / 100.0
        #torch.nn.init.normal_(self.net[-1].weight, 0, 1e-5)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, moving, fixed, features=None):
        theta = torch.Tensor([[[1,0,0],[0,1,0]]]).to(fixed, non_blocking=True)
        grid = torch.nn.functional.affine_grid( \
                theta, moving[0:1].shape, align_corners=False)
        offset = \
                self.net(torch.cat([moving, fixed], 1)).permute(0, 2, 3, 1)
        
        grid = grid + offset
        return grid, offset

    def warp(self, img, grid, interp=False):
        warped = torch.nn.functional.grid_sample( \
                img.float(), grid.float(), align_corners=False)
        if interp and (warped.shape != img.shape):
            warped = torch.nn.functional.interpolate( \
                    warped, size=img.shape[2:])
        return warped

def gradient_loss(s):
    assert s.shape[-1] == 2, 'not 2D grid?'
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dy = torch.abs(s[:, 1:, :, :] - s[:, :-1, :, :])
    dy = dy*dy
    dx = dx*dx
    d = torch.mean(dx)+torch.mean(dy)
    return d/2.0
    
class SpatialTransformer_cohead(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.net = torch.nn.Sequential( \
                UNet(channels, 32, (32, 64, 64, 64, 64)), \
                torch.nn.LeakyReLU(inplace=True), \
                torch.nn.Conv2d(32, 2, kernel_size=3, padding=1))
        with torch.no_grad():
            for param in self.net.parameters():
                param = param / 100.0
        #torch.nn.init.normal_(self.net[-1].weight, 0, 1e-5)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        
    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w = x.shape

        return x.view(b * c, 1, h, w), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w)
    
    def forward(self, moving):      
        target,b1 = self.chans_to_batch_dim(moving[:,:1]).repeat(moving.shape[1]-1,1,1,1)
        moving,b2 = self.chans_to_batch_dim(moving[:,1:])
        
        theta = torch.Tensor([[[1,0,0],[0,1,0]]]).to(moving, non_blocking=True)
        grid = torch.nn.functional.affine_grid(theta, moving[0:1].shape, align_corners=False)
        
        offset = self.net(torch.cat([fix, moving], 1)).permute(0, 2, 3, 1)
        grid = grid + offset
        
        moving = self.warp(moving,grid)
        moving = torch.cat([self.batch_chans_to_chan_dim(target,b1),self.batch_chans_to_chan_dim(moving,b2)],dim=1)
        return moving

    def warp(self, img, grid, interp=False):
        warped = torch.nn.functional.grid_sample( \
                img.float(), grid.float(), align_corners=False)
        if interp and (warped.shape != img.shape):
            warped = torch.nn.functional.interpolate( \
                    warped, size=img.shape[2:])
        return warped

class CatSequential(torch.nn.Module):
    def __init__(self, *modules, dim=1):
        super().__init__()
        self.module = torch.nn.Sequential(*modules)
        self.dim = dim

    def forward(self, x):
        return torch.cat([self.module(x), x], self.dim)

class ResSequential(torch.nn.Module):
    def __init__(self, *modules, sample=None):
        super().__init__()
        self.subnet = torch.nn.Sequential(*modules)
        self.sample = sample

    def forward(self, x):
        out = self.subnet(x)
        x = self.sample(x) if self.sample is not None else x
        return x + out

class NullModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, layers, norm = NullModule):
        super().__init__()
        layers = list(layers)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        act = partial(torch.nn.LeakyReLU, inplace=True)
        #norm = partial(torch.nn.BatchNorm2d) 
        norm = NullModule 
        conv = partial(torch.nn.Conv2d, \
                kernel_size=kernel_size, padding=padding)
        conv_norm_act = lambda in_ch, out_ch: \
                torch.nn.Sequential(conv(in_ch, out_ch), norm(out_ch), act())
        resblock = lambda channels: \
                ResSequential(*[conv_norm_act(channels, channels) \
                for _ in range(num_convs)])
        down = partial(torch.nn.AvgPool2d, kernel_size=2, stride=2)

        # uppest (change channels, resblock)
        current_layer = layers[0]
        self.encoders = torch.nn.ModuleList([torch.nn.Sequential( \
                conv_norm_act(in_channels, current_layer), \
                resblock(current_layer))])
        # middle (change size, change channels, resblock)
        for layer in layers[1:-1]:
            current_layer, upper_layer = layer, current_layer
            self.encoders.append(torch.nn.Sequential( \
                    down(), conv_norm_act(upper_layer, current_layer), \
                    resblock(current_layer)))
        # lowest (change shape and channels only)
        self.encoders.append(torch.nn.Sequential( \
                down(), conv_norm_act(layers[-2], layers[-1])))

    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        return features

class Decoder(torch.nn.Module):
    def __init__(self, out_channels, layers, bridges, norm = NullModule):
        super().__init__()
        layers = list(layers)
        bridges = list(bridges)
        assert len(layers) == len(bridges)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        act = partial(torch.nn.LeakyReLU, inplace=True)
        #norm = partial(torch.nn.BatchNorm2d) 
        conv = partial(torch.nn.Conv2d, \
                kernel_size=kernel_size, padding=padding)
        conv_norm_act = lambda in_ch, out_ch: \
                torch.nn.Sequential(conv(in_ch, out_ch), norm(out_ch), act())
        resblock = lambda channels: \
                ResSequential(*[conv_norm_act(channels, channels) \
                for _ in range(num_convs)])
        up = partial(torch.nn.Upsample, scale_factor=(2, 2))

        # lowest (change channels, resblock, change shape)
        bridge, current_layer = bridges[-1], layers[-1]
        self.decoders = torch.nn.ModuleList([torch.nn.Sequential( \
                conv_norm_act(bridge, current_layer), \
                resblock(current_layer), up())])
        # middle (concatenate*, change channels, resblock, change shape)
        for bridge, current_layer, lower_layer in \
                zip(bridges[-2:0:-1], layers[-2:0:-1], layers[:1:-1]):
            self.decoders.append(torch.nn.Sequential( \
                    conv_norm_act(bridge+lower_layer, current_layer), \
                    resblock(current_layer), up()))
        # uppest (concatenate*, change channels, resblock, change channels)
        bridge, current_layer, lower_layer = bridges[0], layers[0], layers[1]
        self.decoders.append(torch.nn.Sequential( \
                conv_norm_act(bridge+lower_layer, current_layer), \
                resblock(current_layer),
                conv(current_layer, out_channels)))

    def forward(self, bridges):
        x = torch.tensor((), dtype=bridges[0].dtype).to(bridges[0])
        for decoder, bridge in zip(self.decoders, bridges[::-1]):
            x = torch.cat([x, bridge], dim=1)
            x = decoder(x)
        return x

def Conv2d(in_channels, out_channels):
    kernel_size = 3
    padding = kernel_size // 2
    return torch.nn.Sequential( \
            torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))

def Up(in_channels, out_channels):
    return torch.nn.Sequential( \
            torch.nn.Upsample(scale_factor=(2,2)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))

def Down(in_channels, out_channels):
    return torch.nn.Sequential( \
            torch.nn.AvgPool2d(2, stride=2),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=True))



class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super().__init__()
        layers = list(layers)
        kernel_size = 3
        padding = kernel_size // 2
        num_convs = 2
        current_layer = layers.pop()
        upper_layer = layers.pop()
        unet = CatSequential( \
                Down(upper_layer, current_layer), \
                ResSequential( \
                *[Conv2d(current_layer, current_layer) \
                for _ in range(num_convs)]), \
                Up(current_layer, current_layer))
        for layer in reversed(layers):
            lower_layer, current_layer, upper_layer = \
                    current_layer, upper_layer, layer
            unet = CatSequential( \
                    Down(upper_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs)]), \
                    unet, \
                    Conv2d(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    Up(current_layer, current_layer))
        lower_layer, current_layer = \
                current_layer, upper_layer
        self.unet = torch.nn.Sequential( \
                    Conv2d(in_channels, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    unet, \
                    Conv2d(current_layer+lower_layer, current_layer), \
                    ResSequential( \
                    *[Conv2d(current_layer, current_layer) \
                    for _ in range(num_convs-1)]), \
                    torch.nn.Conv2d(current_layer, out_channels, \
                    kernel_size, padding=padding))

    def forward(self, x):
        return self.unet(x)


def conv3x3(in_channels, out_channels):
    kernel_size = 3
    padding = kernel_size // 2
    return torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding)

def conv1x1(in_channels, out_channels):
    kernel_size = 1
    padding = 0
    return torch.nn.Conv2d(in_channels, out_channels, \
            kernel_size, padding=padding)

def ResNet(in_channels, out_channels, channels=[64]*4, res=False):
    net = []
    last = channels[0]
    for current in channels[1:]:
        sample = conv1x1(last, current) if last!=current else None
        net = net + [torch.nn.LeakyReLU(inplace=True), \
                ResSequential( \
                    conv3x3(last, current), \
                    torch.nn.LeakyReLU(inplace=True), \
                    conv3x3(current, current), \
                    sample=sample)]
        last = current
    if res:
        sample = conv1x1(channels[0], channels[-1]) \
                if channels[0]!=channels[-1] else None
        net = [ResSequential(*net, sample=sample)]
    net = [conv3x3(in_channels, channels[0]), \
            *net, \
            torch.nn.LeakyReLU(inplace=True),
            conv3x3(channels[-1], out_channels)]
    return torch.nn.Sequential(*net)