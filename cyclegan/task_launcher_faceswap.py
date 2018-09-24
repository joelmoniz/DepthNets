import numpy as np
import torch
import os
import pickle
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from i2i.cyclegan import CycleGAN
from util import (convert_to_rgb,
                  H5Dataset)
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import rescale, resize

def get_face_swap_iterators(bs):
    """DepthNet + GT <-> frontal GT faces"""
    filename = "faceswap.h5"
    dd_train_a = H5Dataset('data/%s' % filename, 'X_train')
    dd_train_b = H5Dataset('data/%s' % filename, 'Y_train')
    dd_valid_a = H5Dataset('data/%s' % filename, 'X_valid')
    dd_valid_b = H5Dataset('data/%s' % filename, 'Y_valid')
    loader_train_a = DataLoader(dd_train_a, batch_size=bs, shuffle=True)
    loader_train_b = DataLoader(dd_train_b, batch_size=bs, shuffle=True)
    loader_valid_a = DataLoader(dd_valid_a, batch_size=bs, shuffle=True)
    loader_valid_b = DataLoader(dd_valid_b, batch_size=bs, shuffle=True)
    return loader_train_a, loader_train_b, loader_valid_a, loader_valid_b

def image_dump_handler(out_folder, scale_factor=1.):
    def _fn(losses, inputs, outputs, kwargs):
        if kwargs['iter'] != 1:
            return
        A_real = inputs[0].data.cpu().numpy()
        B_real = inputs[1].data.cpu().numpy()
        atob, atob_btoa, btoa, btoa_atob = \
            [elem.data.cpu().numpy() for elem in outputs.values()]
        outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
        # determine # of channels
        n_channels = outs_np[0].shape[1]
        w, h = outs_np[0].shape[-1], outs_np[0].shape[-2]
        # possible that A_real.bs != B_real.bs
        bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
        grid = np.zeros((h*bs, w*6, 3))
        for j in range(bs):
            for i in range(6):
                n_channels = outs_np[i][j].shape[0]
                img_to_write = convert_to_rgb(outs_np[i][j], is_grayscale=False)
                grid[j*h:(j+1)*h, i*w:(i+1)*w, :] = img_to_write
        imsave(arr=rescale(grid, scale=scale_factor),
               fname="%s/%i_%s.png" % (out_folder, kwargs['epoch'], kwargs['mode']))
    return _fn

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str,
                        default="my_experiment")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', choices=['train', 'test', 'vis'],
                        default='train')
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

if __name__ == '__main__':

    from torchvision.utils import save_image

    from architectures import block9_a3b3

    gen_atob, disc_a, gen_btoa, disc_b = block9_a3b3.get_network()

    print("Loading iterators...")
    it_train_a, it_train_b, it_valid_a, it_valid_b = \
        get_face_swap_iterators(args.batch_size)

    print("Loading CycleGAN...")
    name = args.name
    net = CycleGAN(
        gen_atob_fn=gen_atob,
        disc_a_fn=disc_a,
        gen_btoa_fn=gen_btoa,
        disc_b_fn=disc_b,
        opt_d_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        opt_g_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        handlers=[image_dump_handler("%s/%s" % (args.save_path, name))],
        use_cuda=False if args.cpu else True
    )
    if args.resume is not None:
        net.load(args.resume)
    if args.mode == "train":
        print("Training...")
        net.train(
            itr_a_train=it_train_a,
            itr_b_train=it_train_b,
            itr_a_valid=it_valid_a,
            itr_b_valid=it_valid_b,
            epochs=1000,
            model_dir="%s/%s" % (args.model_save_path, name),
            result_dir="%s/%s" % (args.save_path, name),
            append=True if args.resume is not None else False
        )
    elif args.mode == "vis":
        print("Converting A -> B...")
        net.g_atob.eval()
        aa = iter(it_train_a).next()[0]
        bb = net.g_atob(aa)
        save_image(aa*0.5 + 0.5, "tmp/aa.png")
        save_image(bb*0.5 + 0.5, "tmp/bb.png")
    elif args.mode == 'test':
        print("Dropping into pdb...")
        import pdb
        pdb.set_trace()
