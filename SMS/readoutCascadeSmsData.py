'''
MIT License
Copyright (c) 2023 Breeze.
'''
import numpy as np
import sigpy
def readoutCascadeSmsData(kspace_data,factor):
    """
    This function convert a MB factor data set into a inplane parallel
    imaging problem by cascade in readout direction
    input
    return [nCoil,nFE*factor,nPE]
    """
    nCoil, nFE, nPE, = kspace_data.shape
    pseudoPi_data = np.zeros((nCoil,nFE*factor,nPE),dtype=kspace_data.dtype)
    pseudoPi_data[:,::factor,:] = kspace_data
    return pseudoPi_data

def readoutCascadeRestoreReconSlices(readoutCascadeRecon_xy,CAIPI_pattern):
    """
    restore slice to original locations out of immediate readout-cascaded SMS
    recon.
    input [nslice*nFE, nPE],[]
    return [nslice,nFE, nPE]
    """
    nPE = readoutCascadeRecon_xy.shape[-1]
    factor = len(CAIPI_pattern)
    nFE = int(readoutCascadeRecon_xy.shape[-2]/factor)
    restoredSlice_xyz = np.zeros((nFE,nPE,factor),dtype=readoutCascadeRecon_xy.dtype)#
    if factor%2==0:
        readoutCascadeRecon_xy = sigpy.circshift(readoutCascadeRecon_xy, (int(0.5*nFE),0)) 
    readoutCascadeRecon_xy = np.reshape(readoutCascadeRecon_xy, (nFE,factor,nPE), order='F')
    for iSlice in range(factor):
        restoredSlice_xyz[:,:,iSlice]=sigpy.circshift(readoutCascadeRecon_xy[:,iSlice,:],(0,int(1*CAIPI_pattern[iSlice]*nPE)))
    
    return restoredSlice_xyz.transpose((2,0,1))

def SlicesRestore(restoredSlice_xyz,CAIPI_pattern):
    """
    restore slice to original locations out of immediate readout-cascaded SMS
    recon.
    input [nslice,nFE, nPE],[]
    return [nslice,nFE, nPE]
    """
    nPE = restoredSlice_xyz.shape[-1]
    factor = len(CAIPI_pattern)
    nFE = restoredSlice_xyz.shape[-2]
    if factor%2==0:
        restoredSlice_xyz = sigpy.circshift(restoredSlice_xyz, (0,int(0.5*nFE),0)) 
    for iSlice in range(factor):
        restoredSlice_xyz[iSlice,:,:]=sigpy.circshift(restoredSlice_xyz[iSlice,:,:],(0,int(1*CAIPI_pattern[iSlice]*nPE)))
    
    return restoredSlice_xyz

from numpy import pi, e
def del_Phase(kspace_slice_data, shift_Phase):
    nSlice, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(y):
            phase_ramp=e**(1j*2*pi*(Slice)*(i-y//2)*shift_Phase)
            shift_data[Slice,:,i] = kspace_slice_data[Slice,:,i]/phase_ramp
    return shift_data

def readoutCascadeRestoreReconSlices_v2(readoutCascadeRecon_xy,shift_Phase):
    """
    restore slice to original locations out of immediate readout-cascaded SMS
    recon.
    input [nslice*nFE, nPE],[]
    return [nslice,nFE, nPE]
    """
    nPE = readoutCascadeRecon_xy.shape[-1]
    factor = shift_Phase
    nFE = int(readoutCascadeRecon_xy.shape[-2]/factor)
    restoredSlice_xyz = np.zeros((nFE,nPE,factor),dtype=readoutCascadeRecon_xy.dtype)#
    if factor%2==0:
        readoutCascadeRecon_xy = sigpy.circshift(readoutCascadeRecon_xy, (int(-0.5*nFE),0)) 
    readoutCascadeRecon_xy = np.reshape(readoutCascadeRecon_xy, (nFE,factor,nPE), order='F')
    for iSlice in range(factor):
        restoredSlice_xyz[:,:,iSlice]=readoutCascadeRecon_xy[:,iSlice,:]
    restoredSlice_xyz = restoredSlice_xyz.transpose((2,0,1))
    restoredSlice_xyz = del_Phase(restoredSlice_xyz, 1/shift_Phase)
    return restoredSlice_xyz.transpose((2,0,1))
    
