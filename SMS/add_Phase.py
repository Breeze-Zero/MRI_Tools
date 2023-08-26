import numpy as np
from numpy import pi, e

def add_Phase(kspace_slice_data, shift_Phase):
    nSlice, nCoil, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(y):
            phase_ramp=e**(1j*2*pi*(Slice)*(i-y//2)*shift_Phase)
            shift_data[Slice,:,:,i] = kspace_slice_data[Slice,:,:,i]*phase_ramp
    return shift_data

def add_Phase_1(kspace_slice_data, shift_Phase):###这个更像资料里的效果
    nSlice, nCoil, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(x):
            phase_ramp=e**(1j*2*pi*(Slice)*(i-x//2)*shift_Phase)
            shift_data[Slice,:,i,:] = kspace_slice_data[Slice,:,i,:]*phase_ramp
    return shift_data

def del_Phase(kspace_slice_data, shift_Phase):
    nSlice, nCoil, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(y):
            phase_ramp=e**(1j*2*pi*(Slice)*(i-y//2)*shift_Phase)
            shift_data[Slice,:,:,i] = kspace_slice_data[Slice,:,:,i]/phase_ramp
    return shift_data