'''
MIT License
Copyright (c) 2023 Breeze.
'''
import numpy as np
import sigpy
def readoutCascadeAdjustCSM(CSM_xyzc, CAIPI_pattern):
    nFE,nPE,MB_factor,nCoil=CSM_xyzc.shape
    adjustedCSM_xyzc=np.zeros((nFE,MB_factor,nPE,nCoil),dtype=CSM_xyzc.dtype)
    for iSlice in range(MB_factor):
        adjustedCSM_xyzc[:,iSlice,:,:]=sigpy.circshift(CSM_xyzc[:,:,iSlice,:],(0,int(CAIPI_pattern[iSlice]*nPE),0))
        
    adjustedCSM_xyzc = np.reshape(adjustedCSM_xyzc,(nFE*MB_factor,nPE,nCoil), order='F');
    if MB_factor%2==0:
        adjustedCSM_xyzc = sigpy.circshift(adjustedCSM_xyzc,(int(nFE*0.5*1),0,0))
    return adjustedCSM_xyzc
