import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def convert_to_rgb(img, is_grayscale=False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1.
       Acknowledgement: http://github.com/costapt/vess2ret
    """
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))
    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))
    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)
    if not is_grayscale:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.
    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)

class H5Dataset(Dataset):
    def __init__(self, h5_file, key,
                 train=True, train_percent=0.95):
        f = h5py.File(h5_file, 'r')
        self.f = f
        self.key = key
        arr = f[key]
        if train:
            arr = arr[0:int(train_percent*len(arr))]
        else:
            arr = arr[int(train_percent*len(arr))::]
        self.arr = arr
        
    def __getitem__(self, index):
        img = self.arr[index]
        img = ((img / 255.) - 0.5) / 0.5
        img = img.swapaxes(2, 1).swapaxes(1, 0)
        return torch.from_numpy(img).float()

    def __len__(self):
        return self.arr.shape[0]
