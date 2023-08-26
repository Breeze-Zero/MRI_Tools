import torch
import h5py
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import os
import xml.etree.ElementTree as etree
import sys
import fastmri
from fastmri.data.transforms import to_tensor
import MRI_Tools.mri_utils.transforms_numpy as mrfft
from MRI_Tools.SMS.add_Phase import add_Phase as add_Phase
import random
import numpy
import pickle
random.seed(42)
import time
from fastmri.data.subsample import RandomMaskFunc,EquiSpacedMaskFunc
from fastmri.data import transforms as T
#from grappa_cupy import grappa
#from slicegrappa_cupy import slicegrappa
DEBUG = False
def normalize_instance(data, eps= 0.0):
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + eps)

def normalize_max(data, max_value):
    data_norm = data/ max_value
    return data_norm

def center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]

def slice_to_SMS(slice_kspace,crop_size,max_value=1,
                 MB_factor=4,shift_fov = None,norm=False,add_phase=False,only_sms = False):
    """
    input (slice, ncoil, nFE, nPE), (crop_size1, crop_size2)
    return (slice, nCoil, nFE, nPE), (nCoil, nFE, nPE)  # changed by mengye, 20221028
    """
    assert slice_kspace.shape[0] == MB_factor
    if shift_fov is None:
        shift_fov = 1/MB_factor
    temp_img = center_crop(mrfft.ifft2c(slice_kspace),crop_size)#→(slice, coil, crop_size1, crop_size2)
    if MB_factor%2==0 and only_sms:
        temp_img = np.roll(temp_img,-temp_img.shape[2]//2,2)
        
    slice_kspace = mrfft.fft2c(temp_img)    
        
    if add_phase:
        slice_kspace = add_Phase(slice_kspace, shift_fov)
    if norm:
        slice_kspace = normalize_instance(slice_kspace)#normalize_max(slice_kspace,max_value)#
    multislice_data = slice_kspace # changed by mengye, 20221028
    SMS_data = np.sum(multislice_data,0)/multislice_data.shape[0] # changed by mengye, 20221028
    
    return multislice_data,SMS_data # changed by mengye, 20221028

def get_calib_slicegrappa(multislice_data,pd=30):
    """
    input (nFE, nPE, slice, nCoil),pd
    return (2pd, 2pd, nCoil, slice)
    """
    nFE, nPE, nSlice, nCoil = multislice_data.shape
    calib_slicegrappa = multislice_data[nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd, :,:].copy()##centre crop
    calib_slicegrappa = np.moveaxis(calib_slicegrappa,3,2)### change to [nFE, nPE,  nCoil, nSlice]
    return calib_slicegrappa

def SMS_data_to_readout_concat_data(multislice_data, SMS_data, MB_factor=4, pd=30):
    """
    input (slice, ncoil, nFE, nPE), (ncoil, nFE, nPE)
    return (nCoil, nFE*MB_factor, nPE), (nCoil, pd*2, pd*2)
    changed by mengye 20221028
    """
    nCoil, nFE, nPE = SMS_data.shape
    SMS_data_roc = np.zeros((nCoil, nFE*MB_factor, nPE), dtype=multislice_data.dtype)
    SMS_data_roc[:, 0::MB_factor, :] = SMS_data
    #print('multislice_data',np.max(multislice_data)) ###(0.002769542-0.00071215443j)
    temp_img = mrfft.ifftc(multislice_data, -2) 
    #print('temp_img',np.max(temp_img))   ###(3.8724482e-07+1.7119339e-07j)
    
    full_img_roc = np.zeros((nCoil, nFE*MB_factor, nPE), dtype=temp_img.dtype)
    for iSlice in range(MB_factor):
        full_img_roc[:, iSlice*nFE:(iSlice+1)*nFE, :] = temp_img[iSlice,]

    if MB_factor%2 == 0:
        full_img_roc = np.roll(full_img_roc, -nFE//2, 1)
        
        
    full_kspace_roc = mrfft.fftc(full_img_roc, -2)  
    calib = full_kspace_roc[:, 
                        nFE*MB_factor//2-pd:nFE*MB_factor//2+pd, 
                        nPE//2-pd:nPE//2+pd]

    
    return SMS_data_roc, calib

def get_target(slice_target_data,MB_factor=4,shift_fov = 4):
    kspace_slice_data = mrfft.fft2c(slice_target_data)
    
    nSlice, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(y):
            phase_ramp=np.e**(1j*2*np.pi*(Slice)*(i-y//2)*(1/shift_fov))
            shift_data[Slice,:,i] = kspace_slice_data[Slice,:,i]*phase_ramp
    temp_img = mrfft.ifft2c(shift_data)
    full_img_roc = np.zeros((x*MB_factor, y), dtype=temp_img.dtype)
    for iSlice in range(MB_factor):
        full_img_roc[iSlice*x:(iSlice+1)*x] = temp_img[iSlice,:,:]

    if MB_factor%2==0:
        full_img_roc = np.roll(full_img_roc,-x//2,0)
    return np.abs(full_img_roc)
    
def get_sms_target(slice_target_data,MB_factor=4,shift_fov = 4):
    kspace_slice_data = mrfft.fft2c(slice_target_data)
    
    nSlice, x, y=kspace_slice_data.shape
    shift_data = np.zeros(kspace_slice_data.shape,dtype=kspace_slice_data.dtype)
    for Slice in range(nSlice):
        for i in range(y):
            phase_ramp=np.e**(1j*2*np.pi*(Slice)*(i-y//2)*(1/shift_fov))
            shift_data[Slice,:,i] = kspace_slice_data[Slice,:,i]*phase_ramp
    temp_img = mrfft.ifft2c(shift_data)
    if MB_factor%2==0:
        temp_img = np.roll(temp_img,-x//2,1)
    return np.abs(temp_img)


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)

        
class SliceDataset_roc(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                self.examples.append((fname, metadata)) ## 不切片
                # self.examples += [
                #     (fname, slice_ind, metadata) for slice_ind in range(num_slices)
                # ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                # logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            # logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
        
        # self.examples = self.examples * 2
        # manager = Manager()
        # self.examples = manager.list(self.examples)
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        fname, metadata = self.examples[i]
        if self.transform is None:
            pd = 15
        else:
            pd = random.choice([12,15,20,25,30])
        with h5py.File(fname, "r",libver='latest') as hf:
            kspace = hf["kspace"]#[()]
            target = hf[self.recons_key] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)
        
            MB_factor = random.randint(2,5) if kspace.shape[0]>=5 else random.randint(2,kspace.shape[0])

            crop_szie = (target.shape[2],target.shape[1])
            shift_fov = random.randint(2,MB_factor)
    
            gap = random.randint(0,(kspace.shape[0]//MB_factor-1))
            #numpy.linspace(gap, num_slices-1, self.mb_factor, dtype=int)
            # gap = kspace.shape[0]//MB_factor
            i = random.randint(0,gap-1)
            num_slices = kspace.shape[0]
            index = list(range(num_slices))[i::gap]
            if len(index)>MB_factor:
                index = random.sample(index, MB_factor)
                index.sort()
                # index = index[:MB_factor]
            assert MB_factor == len(index)
            slice_kspace = kspace[index]#np.rot90(kspace[index],axes = (-2, -1))#[i::gap,]
            slice_target = target[index]#np.rot90(target[index],axes = (-2, -1))#[i::gap,]

        assert slice_kspace.shape[0]%MB_factor==0
        multislice_data,SMS_data = slice_to_SMS(slice_kspace,crop_szie,
                                                MB_factor=MB_factor,shift_fov =1/shift_fov,max_value=attrs['max'],norm=False,add_phase=True)
        SMS_data_roc,calibration_data = SMS_data_to_readout_concat_data(multislice_data,SMS_data,MB_factor=MB_factor,pd = pd)
        full_img_roc = get_target(slice_target,MB_factor=MB_factor,shift_fov=shift_fov)
  
        nCoil, nFE, nPE = SMS_data_roc.shape
        _, pdx, pdy = calibration_data.shape
        
        full_SMS_data_roc = SMS_data_roc.copy()
        full_SMS_data_roc[:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd] = calibration_data[:, pdx//2-pd:pdx//2+pd,pdy//2-pd:pdy//2+pd]
        
        # attrs_dict = {}
        # attrs_dict['file_name'] = fname
        # attrs_dict['attrs'] = str(attrs)
        # attrs_dict['slice_index'] = index#list(range(kspace.shape[0]))[i::gap]
        # attrs_dict['MB_factor'] = MB_factor 
        

        SMS_data_roc = to_tensor(SMS_data_roc).to(torch.float32)
        full_img_roc = to_tensor(full_img_roc).to(torch.float32)
        full_SMS_data_roc = to_tensor(full_SMS_data_roc).to(torch.float32)
        acs_mask = torch.ne(SMS_data_roc,0)
 
        attrs = eval(str(attrs))

        max_value =  attrs["max"]
        
        
        center_mask = torch.zeros_like(SMS_data_roc)
        center_mask[:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd,:] = torch.ones(center_mask.shape[0],int(2*pd),int(2*pd),center_mask.shape[-1])
        center_mask=center_mask.to(torch.bool)
        
        
        if self.transform is None:
            return SMS_data_roc,full_img_roc,center_mask,max_value,acs_mask,full_SMS_data_roc,torch.tensor([MB_factor,shift_fov,1])#,torch.tensor([shift_fov])
        else:
            center_fractions=[[1],[0.08], [0.04]]
            accelerations=[[1],[4], [8]]
            choice = random.randint(0,2)
            if choice >0:
                MaskFunc = RandomMaskFunc(center_fractions=center_fractions[choice], accelerations=accelerations[choice])
                SMS_data_roc, mask, _ = T.apply_mask(SMS_data_roc, MaskFunc)
                # SMS_data_roc = SMS_data_roc.permute(0, 2, 1, 3)
                # mask = mask.permute(0, 2, 1, 3)
                # center_mask = center_mask * mask
                acs_mask = acs_mask * mask

                
            return SMS_data_roc,full_img_roc,center_mask.to(torch.bool),max_value,acs_mask.to(torch.bool),full_SMS_data_roc,torch.tensor([MB_factor,shift_fov,accelerations[choice][0]])
        

        
class SliceDataset_roc_val(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform = None,
        acceleration = None,
        center_fractions = None,
        pd = 30,
        mb_factor = None,
        shift_fov = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.acceleration = acceleration
        self.center_fractions = center_fractions
        self.pd = pd
        self.mb_factor = mb_factor
        self.shift_fov = shift_fov
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                if num_slices >=self.mb_factor:
                    gap = num_slices//self.mb_factor#random.randint(0,(num_slices//self.mb_factor-1))
                    for i in range(gap):
                        index = list(range(num_slices))[i::gap]
                        if len(index)>self.mb_factor:
                            for num in range(len(index)-self.mb_factor+1):
                                self.examples.append((fname, metadata,index[num:num+self.mb_factor]))
                        else:
                            self.examples.append((fname, metadata,index)) 
         

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                # logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            # logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
        
        # self.examples = self.examples * 2
        # manager = Manager()
        # self.examples = manager.list(self.examples)
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):        
        fname, metadata, index = self.examples[i]
        pd = self.pd
        with h5py.File(fname, "r",libver='latest') as hf:#,swmr=True
            kspace = hf["kspace"]#[()]
            target = hf[self.recons_key] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)
        
            MB_factor = self.mb_factor
            crop_szie = (target.shape[1],target.shape[2])
            shift_fov = self.shift_fov
            slice_kspace = kspace[index]#[i::gap,]
            slice_target = target[index]#[i::gap,]
        #print(slice_kspace.shape)
        
        assert slice_kspace.shape[0]%MB_factor==0
        multislice_data,SMS_data = slice_to_SMS(slice_kspace,crop_szie,
                                                MB_factor=MB_factor,shift_fov =1/shift_fov,max_value=attrs['max'],norm=False,add_phase=True)
        SMS_data_roc,calibration_data = SMS_data_to_readout_concat_data(multislice_data,SMS_data,MB_factor=MB_factor,pd = pd)
        full_img_roc = get_target(slice_target,MB_factor=MB_factor,shift_fov=shift_fov)
  
        nCoil, nFE, nPE = SMS_data_roc.shape
        _, pdx, pdy = calibration_data.shape
        
        full_SMS_data_roc = SMS_data_roc.copy()
        full_SMS_data_roc[:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd] = calibration_data[:, pdx//2-pd:pdx//2+pd,pdy//2-pd:pdy//2+pd]
        
        # attrs_dict = {}
        # attrs_dict['file_name'] = fname
        # attrs_dict['attrs'] = str(attrs)
        # attrs_dict['slice_index'] = index#list(range(kspace.shape[0]))[i::gap]
        # attrs_dict['MB_factor'] = MB_factor 
        

        SMS_data_roc = to_tensor(SMS_data_roc).to(torch.float32)
        full_img_roc = to_tensor(full_img_roc).to(torch.float32)
        full_SMS_data_roc = to_tensor(full_SMS_data_roc).to(torch.float32)
        acs_mask = torch.ne(SMS_data_roc,0)
 
        attrs = eval(str(attrs))

        max_value =  attrs["max"]
        
        center_mask = torch.zeros_like(SMS_data_roc)
        center_mask[:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd,:] = torch.ones(center_mask.shape[0],int(2*pd),int(2*pd),center_mask.shape[-1])
        center_mask=center_mask.to(torch.bool)
        
        
        if self.transform is None:
            return SMS_data_roc,full_img_roc,center_mask,max_value,acs_mask,full_SMS_data_roc,torch.tensor([MB_factor,shift_fov,1])#,torch.tensor([shift_fov])
        else:
            center_fractions = self.center_fractions
            accelerations = self.acceleration 

            MaskFunc = EquiSpacedMaskFunc(center_fractions=center_fractions, accelerations=accelerations)#RandomMaskFunc
            SMS_data_roc, mask, _ = T.apply_mask(SMS_data_roc, MaskFunc)
            center_mask = center_mask * mask
            acs_mask = acs_mask * mask
                
            return SMS_data_roc,full_img_roc,center_mask.to(torch.bool),max_value,acs_mask.to(torch.bool),full_SMS_data_roc,torch.tensor([MB_factor,shift_fov,accelerations[0]])
        
class SliceDataset_SMS(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                # if num_slices>=4:
                self.examples.append((fname, metadata)) ## 不切片
                # self.examples += [
                #     (fname, slice_ind, metadata) for slice_ind in range(num_slices)
                # ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                # logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            # logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
        
        # self.examples = self.examples * 2
        # manager = Manager()
        # self.examples = manager.list(self.examples)
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf["kspace"].shape[0]
            metadata = {}
        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int): 
        fname, metadata = self.examples[i]
        if self.transform is None:
            pd = 15
        else:
            pd = random.choice([12,15,20,25,30])
        with h5py.File(fname, "r",libver='latest') as hf:
            kspace = hf["kspace"]#[()]
            target = hf[self.recons_key] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)
        
            MB_factor = random.randint(2,5) if kspace.shape[0]>=5 else random.randint(2,kspace.shape[0])

            # crop_szie = (target.shape[1],target.shape[2])
            shift_fov = random.randint(2,MB_factor)#MB_factor#
    
            gap = random.randint(0,(kspace.shape[0]//MB_factor-1))
            # gap = kspace.shape[0]//MB_factor
            i = random.randint(0,gap-1)
            num_slices = kspace.shape[0]
            index = list(range(num_slices))[i::gap]
            if len(index)>MB_factor:
                index = random.sample(index, MB_factor)
                index.sort()
                # index = index[:MB_factor]
            assert MB_factor == len(index)
            if attrs['acquisition'] == 'EPI':
                slice_kspace = kspace[index]
                slice_kspace = slice_kspace[:,:,::-1,::-1]
                slice_target = target[index]
                slice_target = slice_target[:,::-1,::-1]
                crop_szie = (target.shape[1],target.shape[2])
            else:
                slice_kspace = kspace[index]#np.rot90(kspace[index],axes = (-2, -1))#[i::gap,]
                slice_target = target[index]#np.rot90(target[index],axes = (-2, -1))#[i::gap,]
                crop_szie = (target.shape[2],target.shape[1])
            
        assert slice_kspace.shape[0]%MB_factor==0
        multislice_data,SMS_data = slice_to_SMS(slice_kspace,crop_szie,
                                                MB_factor=MB_factor,shift_fov =1/shift_fov,max_value=attrs['max'],norm=False,
                                                add_phase=True,only_sms = True)
        nSlice, nCoil, nFE, nPE = multislice_data.shape
        full_SMS_data = np.zeros(multislice_data.shape,dtype=multislice_data.dtype)
        full_SMS_data[:,:,nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd] = multislice_data[:,:,nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd]
        
        
        full_img_target = get_sms_target(slice_target,MB_factor=MB_factor,shift_fov =shift_fov)
        
        # attrs_dict = {}
        # attrs_dict['file_name'] = fname
        # attrs_dict['attrs'] = str(attrs)
        # attrs_dict['slice_index'] = index#list(range(kspace.shape[0]))[i::gap]
        # attrs_dict['MB_factor'] = MB_factor 
        

        SMS_data = to_tensor(SMS_data).to(torch.float32)
        full_img_target = to_tensor(full_img_target).to(torch.float32)
        full_SMS_data = to_tensor(full_SMS_data).to(torch.float32)
 
        attrs = eval(str(attrs))

        max_value =  attrs["max"]
        
        center_mask = torch.zeros_like(full_SMS_data)
        center_mask[:,:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd,:] = torch.ones(center_mask.shape[0],center_mask.shape[1],int(2*pd),int(2*pd),center_mask.shape[-1])
        center_mask=center_mask.to(torch.bool)
        
        if self.transform is None:
            return SMS_data,full_img_target,center_mask,full_SMS_data,torch.tensor([MB_factor,shift_fov,1]),max_value#,torch.tensor([shift_fov])
        else:
            center_fractions=[[1],[0.08],[0.04]]
            accelerations=[[1],[4],[8]]
            choice = random.randint(0,2)
            if choice >0:
                MaskFunc = EquiSpacedMaskFunc(center_fractions=center_fractions[choice], accelerations=accelerations[choice])
                SMS_data, mask, _ = T.apply_mask(SMS_data, MaskFunc)
                center_mask = center_mask * mask.unsqueeze(1)

            return SMS_data,full_img_target,center_mask.to(torch.bool),full_SMS_data,torch.tensor([MB_factor,shift_fov,accelerations[choice][0]]),max_value
        
        
        
class SliceDataset_SMS_val(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform = None,
        acceleration = None,
        center_fractions = None,
        pd = 30,
        mb_factor = None,
        shift_fov = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        self.acceleration = acceleration
        self.center_fractions = center_fractions
        self.pd = pd
        self.mb_factor = mb_factor
        self.shift_fov = shift_fov
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                if num_slices >=self.mb_factor:
                    gap = num_slices//self.mb_factor#random.randint(0,(num_slices//self.mb_factor-1))
                    for i in range(gap):
                        index = list(range(num_slices))[i::gap]
                        if len(index)>self.mb_factor:
                            for num in range(len(index)-self.mb_factor+1):
                                self.examples.append((fname, metadata,index[num:num+self.mb_factor]))
                        else:
                            self.examples.append((fname, metadata,index)) 
         

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                # logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            # logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [
                ex
                for ex in self.examples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
        
        # self.examples = self.examples * 2
        # manager = Manager()
        # self.examples = manager.list(self.examples)
    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):        
        fname, metadata, index = self.examples[i]
        pd = self.pd
        with h5py.File(fname, "r",libver='latest') as hf:#,swmr=True
            kspace = hf["kspace"]#[()]
            target = hf[self.recons_key] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)
        
        
            MB_factor = self.mb_factor
            crop_szie = (target.shape[1],target.shape[2])
            shift_fov = self.shift_fov

            # gap = random.randint(0,(kspace.shape[0]//MB_factor-1))
            # index = numpy.linspace(gap, kspace.shape[0]-1, MB_factor, dtype=int)
            slice_kspace = kspace[index]#[i::gap,]
            slice_target = target[index]#[i::gap,]
        #print(slice_kspace.shape)
  
        
        assert slice_kspace.shape[0]%MB_factor==0
        multislice_data,SMS_data = slice_to_SMS(slice_kspace,crop_szie,
                                                MB_factor=MB_factor,shift_fov =1/shift_fov,max_value=attrs['max'],norm=False,
                                                add_phase=True,only_sms = True)
        nSlice, nCoil, nFE, nPE = multislice_data.shape
        full_SMS_data = np.zeros(multislice_data.shape,dtype=multislice_data.dtype)
        full_SMS_data[:,:,nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd] = multislice_data[:,:,nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd]
        
        full_img_target = get_sms_target(slice_target,MB_factor=MB_factor,shift_fov =shift_fov)
        
        # attrs_dict = {}
        # attrs_dict['file_name'] = fname
        # attrs_dict['attrs'] = str(attrs)
        # attrs_dict['slice_index'] = index#list(range(kspace.shape[0]))[i::gap]
        # attrs_dict['MB_factor'] = MB_factor 
        

        SMS_data = to_tensor(SMS_data).to(torch.float32)
        full_img_target = to_tensor(full_img_target).to(torch.float32)
        full_SMS_data = to_tensor(full_SMS_data).to(torch.float32)
 
        attrs = eval(str(attrs))

        max_value =  attrs["max"]
        center_mask = torch.zeros_like(full_SMS_data)
        center_mask[:,:, nFE//2-pd:nFE//2+pd, nPE//2-pd:nPE//2+pd,:] = torch.ones(center_mask.shape[0],center_mask.shape[1],int(2*pd),int(2*pd),center_mask.shape[-1])
        center_mask=center_mask.to(torch.bool)
        
        if self.transform is None:
            return SMS_data,full_img_target,center_mask,full_SMS_data,torch.tensor([MB_factor,shift_fov,1]),max_value
        else:
            center_fractions = self.center_fractions
            accelerations = self.acceleration 
            MaskFunc = EquiSpacedMaskFunc(center_fractions=center_fractions, accelerations=accelerations)#RandomMaskFunc
            SMS_data_roc, mask, _ = T.apply_mask(SMS_data_roc, MaskFunc)
            center_mask = center_mask * mask
            acs_mask = acs_mask * mask
                
            return SMS_data_roc,full_img_roc,center_mask.to(torch.bool),max_value,acs_mask.to(torch.bool),full_SMS_data_roc,torch.tensor([MB_factor,shift_fov,accelerations[0]])