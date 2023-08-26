"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing
import pathlib
import time

import bart ## you need use sys to add path of bart 
import numpy as np
import torch


import fastmri
from fastmri import tensor_to_complex_np
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type


class DataTransform(object):
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, split, reg_wt=None, mask_func=None, use_seed=True):
        if split in ("train", "val"):
            self.retrieve_acc = False
            self.mask_func = mask_func
        else:
            self.retrieve_acc = True
            self.mask_func = None

        self.reg_wt = reg_wt
        self.use_seed = use_seed

    def __call__(self, kspace, num_low_frequency):
        """
        Data Transformer that simply returns the input masked k-space data and
        relevant attributes needed for running MRI reconstruction algorithms
        implemented in BART.

        Args:
            masked_kspace (numpy.array): Input k-space of shape (num_coils, rows,
                cols) for multi-coil data or (rows, cols) for single coil
                data.
            num_low_frequency (int): Number of low-resolution lines acquired.

        Returns:
            tuple: tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same
                    shape as kspace.
                reg_wt (float): Regularization parameter.
                num_low_freqs (int): Number of low-resolution lines acquired.
        """
        kspace = T.to_tensor(kspace)

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = T.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        if self.retrieve_acc:
            num_low_freqs = attrs["num_low_frequency"]
        else:
            num_low_freqs = None

        reg_wt = self.reg_wt

        return (masked_kspace, reg_wt, num_low_freqs)


def cs_total_variation(kspace, reg_wt, num_iters, num_low_freqs,sens_maps=None,singlecoil=False):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
    based reconstruction algorithm using the BART toolkit.

    Args:
        kspace (tensor): Input k-space of shape (num_coils, rows,
                cols, 2) for multi-coil data or (rows, cols, 2) for single coil
                data.
        reg_wt (float): Regularization parameter. 0.01/0.001
        num_iters (int).

    Returns:
        np.array: Reconstructed image.
    """
    if torch.is_tensor(kspace):
        if singlecoil:
            kspace = kspace.unsqueeze(0)
        kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
        kspace = tensor_to_complex_np(kspace)
    else:
        if singlecoil:
            kspace = kspace[None,]
        kspace = kspace.transpose(1, 2, 0)[None,]

    # estimate sensitivity maps
    if sens_maps is None:
        if num_low_freqs is None:
            sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)
        else:
            sens_maps = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)
    else:
        if torch.is_tensor(sens_maps):
            sens_maps = sens_maps.permute(1, 2, 0, 3).unsqueeze(0)
            sens_maps = tensor_to_complex_np(sens_maps)
        else:
            sens_maps = sens_maps.transpose(1, 2, 0)[None,]

    # use Total Variation Minimization to reconstruct the image
    pred = bart.bart(
        1, f"pics -d0 -S -R T:7:0:{reg_wt} -i {num_iters}", kspace, sens_maps
    )
    pred = torch.from_numpy(np.abs(pred[0]))

    return pred







# with multiprocessing.Pool(args.num_procs) as pool:
#     outputs = pool.map(run_model, range(len(dataset)))





