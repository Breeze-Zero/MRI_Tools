import bart ## you need use sys to add path of bart 
import numpy as np


def ESPIRIT_bart(kspace, num_low_freqs):
    """
    Run ESPIRIT coil sensitivity estimation using the BART toolkit.

    Args:
        kspace (numpy): Input k-space of shape (num_coils, rows,
                cols) for multi-coil data
    Returns:
        np.array: sensitivity maps.
    """


    kspace = kspace.transpose(1, 2, 0)[None,]
    # estimate sensitivity maps
    if num_low_freqs is None:
        sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)
    else:
        sens_maps = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)
    return sens_maps[0]
