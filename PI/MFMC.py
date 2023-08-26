import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms
from MRI_Tools.SME.ESPIRIT_torch import EspiritCalibration
from MRI_Tools.SME.BaseSensitivityModel import SensitivityModel

class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        model
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = model

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, T, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 1, 5, 2, 3, 4).reshape(b, T, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, T, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, T, 2, c, h, w).permute(0, 1, 3, 4, 5, 2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, t, c, h, w = x.shape
        x = x.view(b, t, 2, c // 2 * h * w)

        mean = x.mean(dim=3).view(b, t, 2, 1)
        std = x.std(dim=3).view(b, t, 2, 1)
        x = (x - mean) / std
        x = x.view(b, t, c, h, w).contiguous()

        return x, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        
        b, t, c, h, w = x.shape
        x = x.contiguous().view(b, t, 2, c // 2 * h * w)
        x = x * std + mean
        x = x.view(b, t, c, h, w).contiguous()
        
        
        return x

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        # x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        # x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x




class MFMC_model(nn.Module):
    """
    Multi-frame multi-coil model
    """
    def __init__(
        self,
        model,
        sme_type: str = 'UNet',
        sens_chans: int = 8,
        sens_pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            sme_type: 'UNet'/'ESPIRIT'
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.sme_type = sme_type
        if self.sme_type is 'UNet':
            self.sens_net = SensitivityModel(
                chans=sens_chans,
                num_pools=sens_pools,
                mask_center=mask_center,
            )
        elif self.sme_type is 'ESPIRIT':
            self.sens_net = EspiritCalibration(
                threshold= 0.05,
                kernel_size= 6,
                crop= 0.95,
                max_iter= 100,
            )
        else:
            raise ValueError("sme_type is wrong")

        self.model = NormUnet(model)
        self.dc_weight = nn.Parameter(torch.ones(1))
        
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True
        )

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        """
        masked_kspace : b,T,coil,h,w,2
        mask: b,1,1,h,w,2
        """
 
        b,num_frame,coil,h,w,two = masked_kspace.shape
        masked_kspace = masked_kspace.view(b*num_frame,coil,h,w,two).contiguous()
        
        if self.sme_type is 'UNet':
            sens_maps = self.sens_net(masked_kspace,mask[:,0], num_low_frequencies).view(b,num_frame,coil,h,w,two).contiguous()  
        elif self.sme_type is 'ESPIRIT':
            sens_maps = self.sens_net(masked_kspace,mask[:,0]).view(b,num_frame,coil,h,w,two).contiguous()        
        
        x = self.sens_reduce(masked_kspace,sens_maps) #b,T,1,h,w,2

        x = self.model(x) #b,T,1,h,w,2

        x = self.sens_expand(x,sens_maps) # b,z,coil,h,w,2
        
        zero = torch.zeros(1, 1, 1, 1, 1, 1).to(x)
        soft_dc = torch.where(mask, x - masked_kspace, zero) * self.dc_weight
        
        ## x: 模型生成的
        ## soft_dc：采集的区域中，模型生成-真值的差
        ## 目的： 采集的区域中为真值，零填充部分为模型生成的
        x = x - soft_dc    #masked_kspace - soft_dc - x
        # x = masked_kspace + x * (1-mask)
        
        return x #fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(x)), dim=2) ## b,z,h,w,2
    
    

class MFMC_cascade_model(nn.Module):

    def __init__(
        self,
        model
        sens_chans: int = 8,
        sens_pools: int = 4,
        num_cascades: int = 12,
        mask_center: bool = True,
    ):
        """
        Args:
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sme_type = sme_type
        if self.sme_type is 'UNet':
            self.sens_net = SensitivityModel(
                chans=sens_chans,
                num_pools=sens_pools,
                mask_center=mask_center,
            )
        elif self.sme_type is 'ESPIRIT':
            self.sens_net = EspiritCalibration(
                threshold= 0.05,
                kernel_size= 6,
                crop= 0.95,
                max_iter= 100,
            )
        else:
            raise ValueError("sme_type is wrong")

        '''
        Defining the model in this way seems to share the weights, and if you do not want to share the weights, it is recommended to define the model here rather than the external parameters
        '''
        self.model = nn.ModuleList([NormUnet(model) for _ in range(num_cascades)])
        self.dc_weight = nn.Parameter(torch.ones(1))
        
    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True
        )

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        """
        masked_kspace : b,T,coil,h,w,2
        mask: b,1,1,h,w,2
        """
 
        b,num_frame,coil,h,w,two = masked_kspace.shape
        sens_maps = masked_kspace.view(b*num_frame,coil,h,w,two).contiguous()
        if self.sme_type is 'UNet':
            sens_maps = self.sens_net(sens_maps,mask[:,0], num_low_frequencies).view(b,num_frame,coil,h,w,two).contiguous()  
        elif self.sme_type is 'ESPIRIT':
            sens_maps = self.sens_net(sens_maps,mask[:,0]).view(b,num_frame,coil,h,w,two).contiguous()        
        
        x = masked_kspace.clone()
        for model in self.model:
            x = self.sens_reduce(x,sens_maps) #b,T,1,h,w,2
            x = model(x) #b,T,1,h,w,2
            x = self.sens_expand(x,sens_maps) # b,z,coil,h,w,2
            zero = torch.zeros(1, 1, 1, 1, 1, 1).to(x)
            soft_dc = torch.where(mask, x - masked_kspace, zero) * self.dc_weight
            x = masked_kspace - soft_dc - x
        
        
        return x #fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(x)), dim=2) ## b,z,h,w,2