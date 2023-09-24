from MRI_Tools.PI.RAKI.rakiModels import rakiReco
import numpy as np


def RAKI(kspace_zf,acs_size,R_dim = 1,device='cuda:0'):
    '''
    kspace_zf : (coils, number phase encoding lines, number read out lines)
    acs_size : (high,width)
    '''
    if R_dim ==2:
        kspace_zf = np.transpose(kspace_zf,(0,2,1))
        acs_size = (acs_size[1],acs_size[0])
        
    # check k-space scaling such that minimum signal has order of magnitude 0       
    scaling = np.floor(np.log10(np.min(np.abs(kspace_zf[np.where(kspace_zf!=0)]))))
    kspace_zf *= 10**(-1*int(scaling))
    
    pdx = (kspace_zf.shape[1]-acs_size[0])//2
    pdy = (kspace_zf.shape[2]-acs_size[1])//2
    acs = kspace_zf[:, pdx:pdx+acs_size[0],pdy:pdy+acs_size[1]].copy()

    acq = np.where(kspace_zf[0,:,0]!=0)[0] # get index of first non-zero sample
    
    raki_output = kspace_zf.copy()
    kspace_zf = kspace_zf[:,acq[0]:acq[-1],:] # the code does not allow for leading zeros 
    (nC, nP, nR) = kspace_zf.shape # (coils, number phase encoding lines, number read out lines)

    R = acq[1] - acq[0]


    layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                     'input_unit': nC, # number channels in input layer, nC is coil number 
                        1:[256,(2,5)], # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                        2:[128,(1,1)], # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                    'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                    }
    raki_output[:,acq[0]:acq[-1],:] = rakiReco(kspace_zf, acs, R, layer_design_raki,device)
    
    
    raki_output = raki_output/10**(-1*int(scaling))
    
    if R_dim ==2:
        raki_output = np.transpose(raki_output,(0,2,1))

    return raki_output
