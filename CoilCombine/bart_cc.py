import h5py
import os
from tqdm import tqdm
import sys
path = 'bart path'
sys.path.append(path);
from bart import bart
import numpy as np
import math

def VCC(kspace, caxis=0):
    vcc_ksp = kspace.copy()
    vcc_ksp = np.conj(vcc_ksp[:,:,::-1,::-1])
    if vcc_ksp.shape[2] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,2)
    if vcc_ksp.shape[3] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,3)
    out = np.concatenate((kspace,vcc_ksp), axis=caxis)
    return out


def bart_cc(ksp,covert2coil):
    '''
    ksp : (slice,coil,h,w)
    covert2coil : int
    '''
    cc = bart(1,f'cc -p {covert2coil} -A -S',np.moveaxis(ksp,1,-1))
    cc = np.moveaxis(cc,-1,1)
    return cc