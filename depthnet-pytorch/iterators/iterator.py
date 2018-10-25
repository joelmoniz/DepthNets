import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if "DIR_3DFAW" not in os.environ:
    raise Exception("DIR_3DFAW env variable not found -- source env.sh")
DATA_DIR = os.environ["DIR_3DFAW"]

def get_iterator_train(bs):
    ROOT = "%s/train.npz" % DATA_DIR
    dat = np.load(ROOT)
    # ['imgs', 'y_keypts', 'z_keypts', 'x_keypts']
    y_keypts = torch.from_numpy(dat['y_keypts']).float()
    z_keypts = torch.from_numpy(dat['z_keypts']).float()
    #x_keypts = torch.from_numpy(dat['x_keypts']).float()
    ds = TensorDataset(y_keypts, z_keypts)
    data_loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True
    )
    return data_loader

def get_iterator_valid(bs):
    ROOT = "%s/valid.npz" % DATA_DIR
    dat = np.load(ROOT)
    # ['imgs', 'y_keypts', 'z_keypts', 'x_keypts']
    y_keypts = torch.from_numpy(dat['y_keypts']).float()
    z_keypts = torch.from_numpy(dat['z_keypts']).float()
    #x_keypts = torch.from_numpy(dat['x_keypts']).float()
    ds = TensorDataset(y_keypts, z_keypts)
    data_loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True
    )
    return data_loader

def get_data_test():
    ROOT = "%s/test.npz" % DATA_DIR
    dat = np.load(ROOT)
    return dat['y_keypts'], dat['z_keypts'], dat['orientations']   

def get_data_valid():
    ROOT = "%s/valid.npz" % DATA_DIR
    dat = np.load(ROOT)
    return dat['y_keypts'], dat['z_keypts'],

def get_data_train():
    ROOT = "%s/train.npz" % DATA_DIR
    dat = np.load(ROOT)
    return dat['y_keypts'], dat['z_keypts']

def get_iterators(bs):
    itr_train = get_iterator_train(bs)
    itr_valid = get_iterator_valid(bs)
    return itr_train, itr_valid

if __name__ == '__main__':
    test_data = get_data_test()
    import pdb
    pdb.set_trace()
