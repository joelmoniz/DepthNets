import numpy as np
import torch
import glob
import os
import pickle
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.dataset import (TensorDataset,
                                      ConcatDataset)
from i2i.cyclegan import CycleGAN
from util import (convert_to_rgb,
                  H5Dataset,
                  DatasetFromFolder)
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import rescale, resize
from importlib import import_module

def get_face_swap_iterators(bs):
    """DepthNet + GT <-> frontal GT faces"""
    filename_vgg = "data/vgg/vgg.h5"
    filename_celeba = "data/celeba/celebA.h5"
    filename_celeba_swap = "data/celeba_faceswap/celeba_faceswap.h5"
    a_train = H5Dataset(filename_celeba_swap, 'imgs', train=True)
    vgg_side_train = H5Dataset('%s' % filename_vgg, 'src_GT', train=True)
    vgg_frontal_train = H5Dataset('%s' % filename_vgg, 'tg_GT', train=True)
    celeba_side_train = H5Dataset('%s' % filename_celeba, 'src_GT', train=True)
    celeba_frontal_train = H5Dataset('%s' % filename_celeba, 'tg_GT', train=True)
    b_train = ConcatDataset((vgg_side_train,
                             vgg_frontal_train,
                             celeba_side_train,
                             celeba_frontal_train))
    a_valid = H5Dataset(filename_celeba_swap, 'imgs', train=False)
    vgg_side_valid = H5Dataset('%s' % filename_vgg, 'src_GT', train=False)
    vgg_frontal_valid = H5Dataset('%s' % filename_vgg, 'tg_GT', train=False)
    celeba_side_valid = H5Dataset('%s' % filename_celeba, 'src_GT', train=False)
    celeba_frontal_valid = H5Dataset('%s' % filename_celeba, 'tg_GT', train=False)
    b_valid = ConcatDataset((vgg_side_valid,
                             vgg_frontal_valid,
                             celeba_side_valid,
                             celeba_frontal_valid))
    loader_train_a = DataLoader(a_train, batch_size=bs, shuffle=True)
    loader_train_b = DataLoader(b_train, batch_size=bs, shuffle=True)
    loader_valid_a = DataLoader(a_valid, batch_size=bs, shuffle=True)
    loader_valid_b = DataLoader(b_valid, batch_size=bs, shuffle=True)
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

if __name__ == '__main__':

    from torchvision.utils import save_image

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--name', type=str,
                            default="my_experiment")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--network', type=str, default=None)
        parser.add_argument('--mode', choices=['train', 'test', 'vis'],
                            default='train')
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--loss', type=str, choices=['mse', 'bce'],
                            default='mse')
        parser.add_argument('--lamb', type=float, default=10.0)
        parser.add_argument('--beta', type=float, default=0.0)
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

    print("Loading iterators...")
    it_train_a, it_train_b, it_valid_a, it_valid_b = \
        get_face_swap_iterators(args.batch_size)

    print("Loading CycleGAN...")
    name = args.name
    net = CycleGAN(
        gen_atob_fn=gen_atob_fn,
        disc_a_fn=disc_a_fn,
        gen_btoa_fn=gen_btoa_fn,
        disc_b_fn=disc_b_fn,
        loss=args.loss,
        lamb=args.lamb,
        beta=args.beta,
        opt_d_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        opt_g_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        handlers=[image_dump_handler("%s/%s" % (args.save_path, name))],
        use_cuda=False if args.cpu else True
    )
    if args.resume is not None:
        if args.resume == 'auto':
            # autoresume
            model_dir = "%s/%s" % (args.model_save_path, name)
            # List all the pkl files.
            files = glob.glob("%s/*.pkl" % model_dir)
            # Make them absolute paths.
            files = [ os.path.abspath(key) for key in files ]
            if len(files) > 0:
                # Get creation time and use that.
                latest_model = max(files, key=os.path.getctime)
                print("Auto-resume mode found latest model: %s" %
                      latest_model)
                net.load(latest_model)
        else:
            print("Loading model: %s" % args.resume)
            net.load(args.resume)
    if args.mode == "train":
        print("Training...")
        net.train(
            itr_a_train=it_train_a,
            itr_b_train=it_train_b,
            itr_a_valid=it_valid_a,
            itr_b_valid=it_valid_b,
            epochs=args.epochs,
            model_dir="%s/%s" % (args.model_save_path, name),
            result_dir="%s/%s" % (args.save_path, name)
        )
    elif args.mode == "vis":
        print("Converting A -> B...")
        net.g_atob.eval()
        aa = iter(it_train_a).next()[0:1]
        bb = net.g_atob(aa)
        save_image(aa*0.5 + 0.5, "tmp/aa.png")
        save_image(bb*0.5 + 0.5, "tmp/bb.png")
    elif args.mode == 'test':
        print("Dropping into pdb...")
        import pdb
        pdb.set_trace()
