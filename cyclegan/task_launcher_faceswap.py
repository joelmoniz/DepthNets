import numpy as np
import torch
import os
import pickle
import argparse
from torch.utils.data import DataLoader
from christorch.i2i.cyclegan import CycleGAN
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms
from skimage.io import imsave, imread
from skimage.transform import rescale, resize

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

def get_face_swap_iterators(root,
                            batch_size, quick_run=False):
    
    if not quick_run:
        pkl_file = "%s/VGG_celebA.pickle" % root
        pkl_file_faceswap = "%s/multiwarped_images_warped.pickle" % root
        with open(pkl_file_faceswap, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
            X = dataset['buf']
            X_train = X[0:int(X.shape[0]*0.95)]
            X_valid = X[int(X.shape[0]*0.95)::]
        with open(pkl_file, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
            Y = dataset['src_GT'][:, :, :, ::-1]
            Y_train = Y[0:int(Y.shape[0]*0.95)]
            Y_valid = Y[int(Y.shape[0]*0.95)::]
        X_train = ((X_train.swapaxes(3, 2).swapaxes(2, 1) / 255.) - 0.5) / 0.5
        Y_train = ((Y_train.swapaxes(3, 2).swapaxes(2, 1) / 255.) - 0.5) / 0.5
        X_valid = ((X_valid.swapaxes(3, 2).swapaxes(2, 1) / 255.) - 0.5) / 0.5
        Y_valid = ((Y_valid.swapaxes(3, 2).swapaxes(2, 1) / 255.) - 0.5) / 0.5
    else:
        # Load a sample file which is a small subset of the PKL_SINA
        # and PKL_FACESWAP files.
        dat = np.load("%s/get_face_swap_iterators_sample.npz" % root)
        X_train = dat['X_train']
        Y_train = dat['Y_train']
        X_valid = dat['X_valid']
        Y_valid = dat['Y_valid']
    dd_train_a = TensorDataset(torch.from_numpy(X_train).float())
    dd_train_b = TensorDataset(torch.from_numpy(Y_train).float())
    dd_valid_a = TensorDataset(torch.from_numpy(X_valid).float())
    dd_valid_b = TensorDataset(torch.from_numpy(Y_valid).float())
    loader_train_a = DataLoader(dd_train_a,
                                batch_size=batch_size, shuffle=True)
    loader_train_b = DataLoader(dd_train_b,
                                batch_size=batch_size, shuffle=True)
    loader_valid_a = DataLoader(dd_valid_a,
                                batch_size=batch_size, shuffle=True)
    loader_valid_b = DataLoader(dd_valid_b,
                                batch_size=batch_size, shuffle=True)
    return loader_train_a, loader_train_b, loader_valid_a, loader_valid_b

def cg_dump_vis(out_folder, scale_factor=1.):
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
    parser.add_argument('--data_dir', type=str,
                        default="/data/milatmp1/beckhamc/tmp_data/joel_faces/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--quick_dataset', action='store_true')
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

    from architectures.networks import define_G
    from architectures.image2image_old import Discriminator
    from torchvision.utils import save_image

    print("Loading iterators...")
    it_train_a, it_train_b, it_valid_a, it_valid_b = \
        get_face_swap_iterators(args.data_dir,
                                args.batch_size,
                                quick_run=args.quick_dataset)

    print("Loading CycleGAN...")
    name = args.name
    net = CycleGAN(
        gen_atob_fn=define_G(
            **{'input_nc': 3, 'ngf': 64, 'output_nc': 3, 'which_model_netG': 'resnet_9blocks', 'norm': 'instance'}),
        disc_a_fn=Discriminator(
            **{'input_dim': 3, 'num_filter': 64, 'output_dim': 1}),
        gen_btoa_fn=define_G(
            **{'input_nc': 3, 'ngf': 64, 'output_nc': 3, 'which_model_netG': 'resnet_9blocks', 'norm': 'instance'}),
        disc_b_fn=Discriminator(
            **{'input_dim': 3, 'num_filter': 64, 'output_dim': 1}),
        opt_d_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        opt_g_args={'lr': args.lr, 'betas': (args.beta1, args.beta2)},
        handlers=[cg_dump_vis("%s/%s" % (args.save_path, name))],
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
