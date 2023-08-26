import sigpy.mri as mr
import numpy as np


def ESPIRIT_sigpy(kspace, num_low_freqs,device=0):
    """
    Run ESPIRIT coil sensitivity estimation using the BART toolkit.

    Args:
        kspace (numpy): Input k-space of shape (num_coils, rows,
                cols) for multi-coil data
    Returns:
        np.array: sensitivity maps.
    """

    # estimate sensitivity maps
    sens_maps = mr.app.EspiritCalib(kspace,calib_width=num_low_freqs,thresh=0.02, kernel_width=6,
                                    crop=0.9,max_iter=100, device=device,
                                    output_eigenvalue=False, show_pbar=False).run().get()

    return sens_maps
