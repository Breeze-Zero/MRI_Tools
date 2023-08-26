import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import cv2
import random
from cv2 import rotate
import math
import sys
import MRI_Tools.mri_utils.transforms_numpy as mrfft_fast
from torch.utils.data import Dataset
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
import torch
import mat73
def normalize_max(data):
    # percentage_005 = np.percentile(data,0.1)    
    # data = np.clip(data,percentage_005,np.max(data))
    data_norm = (data-np.min(data))/ (np.max(data)-np.min(data))    
    return data_norm   

def zpad(array_in, outshape):    
    c,s,x,y = array_in.shape
    oldshape = (x,y)
    assert len(oldshape)==len(outshape)
    pad_list=[(0,0),(0,0)]
    for iold, iout in zip(oldshape, outshape):
        left = math.floor((iout-iold)/2)
        right = math.ceil((iout-iold)/2)
        if left<0 or right<0:
            left = 0
            right = 0
        pad_list.append((left, right))
    zfill = np.pad(array_in, pad_list, 'constant')                     # fill blade into square with 0  
    return zfill

def normal(image):
    percentage_005 = np.percentile(image,0.5)
    percentage_095 = np.percentile(image,99.5)
    mean = np.mean(image)
    std = np.std(image)
    image = np.clip(image,percentage_005,percentage_095)
    image = (image-mean)/(std+1e-8)
    return image

class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""
    def __init__(self, path_list,noise_type='gaussian',noise_param = 50,transform=None):
        super(AbstractDataset, self).__init__()

        self.path_list = path_list
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.transform = transform
    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        c, w, h = img.shape
 
        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (c, w, h))
            # Add noise and clip
            noise_img = img + noise/255

        noise_img = np.clip(noise_img, 0, 1)
        return noise_img
    
    def __len__(self):
        """Returns length of dataset."""

        return len(self.path_list)
    
    def __getitem__(self, index):
        x_path, y_path = self.path_list[index]
        
        with h5py.File(x_path, "r") as hfx:
            x = hfx['reconstruction_rss']
            slic = random.randint(0,x.shape[0]-1)
            x = x[slic:slic+1]
            x = normalize_max(x)
        
        with h5py.File(y_path, "r") as hfy:
            y = hfy['reconstruction_rss']
            y = y[slic:slic+1]
            y = normalize_max(y)  
        
        if self.transform:
            x = x.transpose(1, 2, 0)
            y = y.transpose(1, 2, 0)
            transformed = self.transform(image=x,mask=y)
            x,y = transformed['image'].transpose(2,0,1),transformed['mask'].transpose(2,0,1)

            x_noise = torch.from_numpy(self._add_noise(x)).to(torch.float32)
            x_base = torch.from_numpy(x).to(torch.float32)
            y_noise = torch.from_numpy(self._add_noise(y)).to(torch.float32)
            y_base = torch.from_numpy(y).to(torch.float32)
            return x_noise, x_base,y_noise, y_base
        else:
            x_noise = torch.from_numpy(self._add_noise(x)).to(torch.float32)
            x_base = torch.from_numpy(x).to(torch.float32)
            # y_noise = torch.from_numpy(self._add_noise(y)).to(torch.float32)
            y_base = torch.from_numpy(y).to(torch.float32)
            return x_noise, x_base,y_base
    
    
def Build_dataset(root_path='/data0/M4RawV1.5'):
    train_image_dict = {}
    val_image_dict = {}
    subject_list = os.listdir(os.path.join(root_path,'multicoil_train'))
    for subject in subject_list:
        if '.h5' not in subject:
            continue
        if subject.split(".")[0][:-2] in train_image_dict:
            train_image_dict[subject.split(".")[0][:-2]].append(os.path.join(root_path,'multicoil_train',subject))
        else:
            train_image_dict[subject.split(".")[0][:-2]] = [os.path.join(root_path,'multicoil_train',subject)]
  
        
               
    subject_list = os.listdir(os.path.join(root_path,'multicoil_val'))

    for subject in subject_list:
        if '.h5' not in subject:
            continue
        if subject.split(".")[0][:-2] in val_image_dict:
            val_image_dict[subject.split(".")[0][:-2]].append(os.path.join(root_path,'multicoil_val',subject))
        else:
            val_image_dict[subject.split(".")[0][:-2]] = [os.path.join(root_path,'multicoil_val',subject)]
    train_image_list = []
    val_image_list = []
    for i in train_image_dict.keys():
        if len(train_image_dict[i])==2:
            train_image_list.append([train_image_dict[i][0],train_image_dict[i][1]])
        else:
            train_image_list.append([train_image_dict[i][0],train_image_dict[i][1]])
            train_image_list.append([train_image_dict[i][0],train_image_dict[i][2]])
      
    for i in val_image_dict.keys():
        if len(val_image_dict[i])==2:
            val_image_list.append([val_image_dict[i][0],val_image_dict[i][1]])
        else:
            val_image_list.append([val_image_dict[i][0],val_image_dict[i][1]])
            val_image_list.append([val_image_dict[i][0],val_image_dict[i][2]])
      
    train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    train_dataset = AbstractDataset(train_image_list,noise_type='gaussian',noise_param = 50,transform=train_transform)
    val_dataset = AbstractDataset(val_image_list,noise_type='gaussian',noise_param = 50,transform=None)
    
    return train_dataset,val_dataset
