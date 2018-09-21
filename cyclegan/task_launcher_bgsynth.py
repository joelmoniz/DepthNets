import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from i2i.cyclegan import CycleGAN
from skimage.transform import rescale
from skimage.io import imsave
from torch.utils.data import Dataset
import h5py
from util import convert_to_rgb

from importlib import import_module

VALID_DATASETS = ['depthnet_bg_vs_frontal',
                  'gt_vs_frontal_gt',
                  'depthnet_gt_vs_frontal']

class H5Dataset(Dataset):
    def __init__(self, h5_file, key):
        f = h5py.File(h5_file, 'r')
        self.f = f
        self.key = key
        
    def __getitem__(self, index):
        img = self.f[self.key][index]
        img = ((img / 255.) - 0.5) / 0.5
        img = img.swapaxes(2, 1).swapaxes(1, 0)
        return torch.from_numpy(img).float()

    def __len__(self):
        return self.f[self.key].shape[0]

#####################
# UTILITY FUNCTIONS #
#####################

def image_dump_handler(out_folder, scale_factor=1.):
    """
    These are for images where A is a 6-channel and B is a 3-channel.
    """
    def _fn(losses, inputs, outputs, attrs):
        A_real = inputs[0].cpu().numpy()
        B_real = inputs[1].cpu().numpy()
        atob = outputs['atob'].cpu().numpy()
        atob_btoa = outputs['atob_btoa'].cpu().numpy()
        btoa = outputs['btoa'].cpu().numpy()
        btoa_atob = outputs['btoa_atob'].cpu().numpy()
        outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
        # determine # of channels
        n_channels = outs_np[0].shape[1]
        is_gray = True if n_channels==1 else False
        shp = outs_np[0].shape[-1]
        # possible that A_real.bs != B_real.bs
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((shp*bs, shp*(6+2), 3))
        for j in range(bs):
            grid[j*shp:(j+1)*shp, 0:shp, :] = convert_to_rgb(
                outs_np[0][j][0:3], is_gray) # A_real 0:3
            if outs_np[0][j].shape[0] > 3:
                grid[j*shp:(j+1)*shp, 1*shp:2*shp, :] = convert_to_rgb(
                    outs_np[0][j][3:6], is_gray) # A_real 3:6
            grid[j*shp:(j+1)*shp, 2*shp:3*shp, :] = convert_to_rgb(
                outs_np[1][j][0:3], is_gray) # atob
            grid[j*shp:(j+1)*shp, 3*shp:4*shp, :] = convert_to_rgb(
                outs_np[2][j][0:3], is_gray) # atob_btoa 0:3
            if outs_np[2][j].shape[0] > 3:
                grid[j*shp:(j+1)*shp, 4*shp:5*shp, :] = convert_to_rgb(
                    outs_np[2][j][3:6], is_gray) # atob_btoa 3:6
            grid[j*shp:(j+1)*shp, 5*shp:6*shp, :] = convert_to_rgb(
                outs_np[3][j][0:3], is_gray) # b_real
            grid[j*shp:(j+1)*shp, 6*shp:7*shp, :] = convert_to_rgb(
                outs_np[4][j][0:3], is_gray) # btoa 0:3
            grid[j*shp:(j+1)*shp, 7*shp:8*shp, :] = convert_to_rgb(
                outs_np[5][j][0:3], is_gray) # btoa_atob
        imsave(arr=rescale(grid, scale=scale_factor),
               fname="%s/%i_%s.png" % (out_folder, attrs['epoch'], attrs['mode']))
    return _fn

##################
# DATA ITERATORS #
##################

def get_depthnet_bg_iterators_h5(bs):
    """DepthNet + background <-> frontal GT faces"""
    filename = "depthnet_bg_vs_frontal.h5"
    dd_train_a = H5Dataset('data/%s' % filename, 'X_train')
    dd_train_b = H5Dataset('data/%s' % filename, 'Y_train')
    dd_valid_a = H5Dataset('data/%s' % filename, 'X_valid')
    dd_valid_b = H5Dataset('data/%s' % filename, 'Y_valid')
    loader_train_a = DataLoader(dd_train_a, batch_size=bs, shuffle=True)
    loader_train_b = DataLoader(dd_train_b, batch_size=bs, shuffle=True)
    loader_valid_a = DataLoader(dd_valid_a, batch_size=bs, shuffle=True)
    loader_valid_b = DataLoader(dd_valid_b, batch_size=bs, shuffle=True)
    return loader_train_a, loader_train_b, loader_valid_a, loader_valid_b

def get_gt_iterators_h5(bs):
    """GT <-> frontal GT"""
    filename = "gt_vs_frontal_gt.h5"
    dd_train_a = H5Dataset('data/%s' % filename, 'X_train')
    dd_train_b = H5Dataset('data/%s' % filename, 'Y_train')
    dd_valid_a = H5Dataset('data/%s' % filename, 'X_valid')
    dd_valid_b = H5Dataset('data/%s' % filename, 'Y_valid')
    loader_train_a = DataLoader(dd_train_a, batch_size=bs, shuffle=True)
    loader_train_b = DataLoader(dd_train_b, batch_size=bs, shuffle=True)
    loader_valid_a = DataLoader(dd_valid_a, batch_size=bs, shuffle=True)
    loader_valid_b = DataLoader(dd_valid_b, batch_size=bs, shuffle=True)
    return loader_train_a, loader_train_b, loader_valid_a, loader_valid_b

def get_depthnet_gt_iterators_h5(bs):
    """DepthNet + GT <-> frontal GT faces"""
    filename = "depthnet_gt_vs_frontal.h5"
    dd_train_a = H5Dataset('data/%s' % filename, 'X_train')
    dd_train_b = H5Dataset('data/%s' % filename, 'Y_train')
    dd_valid_a = H5Dataset('data/%s' % filename, 'X_valid')
    dd_valid_b = H5Dataset('data/%s' % filename, 'Y_valid')
    loader_train_a = DataLoader(dd_train_a, batch_size=bs, shuffle=True)
    loader_train_b = DataLoader(dd_train_b, batch_size=bs, shuffle=True)
    loader_valid_a = DataLoader(dd_valid_a, batch_size=bs, shuffle=True)
    loader_valid_b = DataLoader(dd_valid_b, batch_size=bs, shuffle=True)
    return loader_train_a, loader_train_b, loader_valid_a, loader_valid_b

##############
# MAIN STUFF #
##############

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--name', type=str,
                            default="my_experiment")
        parser.add_argument('--dataset',
                            type=str,
                            choices=VALID_DATASETS,
                            default=VALID_DATASETS[0])
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--network', type=str, default=None)
        parser.add_argument('--mode', choices=['train', 'test', 'vis'],
                            default='train')
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--save_path', type=str,
                            default='./results')
        parser.add_argument('--model_save_path', type=str,
                            default='./models')
        parser.add_argument('--cpu', action='store_true')
        args = parser.parse_args()
        return args

    args = parse_args()

    # Dynamically load in the selected generator
    # module.
    mod = import_module(args.network.replace("/", ".").\
                        replace(".py", ""))
    gen_atob_fn, disc_a_fn, gen_btoa_fn, disc_b_fn = mod.get_network()

    def get_dataset(name, bs):
        if name == 'depthnet_bg_vs_frontal':
            it_train_a, it_train_b, it_valid_a, it_valid_b = \
                get_depthnet_bg_iterators_h5(bs)
        elif name == 'gt_vs_frontal_gt':
            it_train_a, it_train_b, it_valid_a, it_valid_b = \
                get_gt_iterators_h5(bs)
        elif name == 'depthnet_gt_vs_frontal':
            it_train_a, it_train_b, it_valid_a, it_valid_b = \
                get_depthnet_gt_iterators_h5(bs)
        else:
            raise Exception("%s is not a valid dataset" % name)
        return it_train_a, it_train_b, it_valid_a, it_valid_b

    it_train_a, it_train_b, it_valid_a, it_valid_b = \
        get_dataset(args.dataset, args.batch_size)

    net = CycleGAN(
        gen_atob_fn=gen_atob_fn,
        disc_a_fn=disc_a_fn,
        gen_btoa_fn=gen_btoa_fn,
        disc_b_fn=disc_b_fn,
        opt_d_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        opt_g_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        handlers=[image_dump_handler("%s/%s" % (args.save_path, args.name))],
        use_cuda='detect',
        beta=0.
    )

    bs = args.batch_size
    if args.mode == "train":
        net.train(
            itr_a_train=it_train_a,
            itr_b_train=it_train_b,
            itr_a_valid=it_valid_a,
            itr_b_valid=it_valid_b,
            epochs=args.epochs,
            model_dir="%s/%s" % (args.model_save_path, args.name),
            result_dir="%s/%s" % (args.save_path, args.name),
        )

    #fn = cg_baseline_sina_newblock9_inorm_sinazoom('import')
    #fn = cg_hardbaseline2_sina_newblock9_inorm_sinazoom('import')
    #fn = cg_baseline_sina_newblock9_inorm_sinazoom_fullbg('import')


