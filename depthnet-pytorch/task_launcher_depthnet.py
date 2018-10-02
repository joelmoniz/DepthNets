import numpy as np
import torch
from torch.autograd import Variable, grad
from christorch.gan.architectures import disc, gen
from christorch.gan.iterators import mnist
from christorch import util
import imp
import argparse
import glob
import os
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from depthnet_gan import DepthNetGAN, zip_iter, save_handler
from depthnet_gan_learnm import DepthNetGAN_M
from depthnet_gan_sdam import DepthNetGAN_SDAM
import interactive

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default="deleteme")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lamb', type=float, default=10.)
    parser.add_argument('--dnorm', type=float, default=0.)
    parser.add_argument('--l2_decay', type=float, default=0.)
    # Iterator returns (it_train_a, it_train_b, it_val_a, it_val_b)
    parser.add_argument('--iterator', type=str, default="iterators/mnist.py")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--interactive', type=str, default=None)
    parser.add_argument('--use_l1', action='store_true')
    parser.add_argument('--cheat', action='store_true',
                        help='Regress ground truth depth')
    parser.add_argument('--no_gan', action='store_true')
    parser.add_argument('--detach', action='store_true',
                        help='Do not backprop through m (secondary ' +
                        'least squares depth estimation')
    spec = parser.add_mutually_exclusive_group()
    spec.add_argument('--learn_m', action='store_true')
    spec.add_argument('--sdam', action='store_true')
    spec.add_argument('--sigma', type=float, default=1.,
                       help='Regularisation term for pseudo-inv')
    parser.add_argument('--update_g_every', type=int, default=1)
    parser.add_argument('--network', type=str, default="networks/mnist.py")
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--save_images_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    return args

def plot_xy_and_z(xy_real, z_real, z_fake,
                  out_file):
    fig = plt.figure(figsize=(8,4))
    # Do the 2D ground truth keypts.
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(xy_real[0],
               xy_real[1],
               z_real)
    ax.view_init(30, 30)
    ax.set_title('real depth')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xy_real[0],
               xy_real[1],
               z_fake)
    ax.view_init(30, 30)
    ax.set_title('pred depth')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    fig.savefig(out_file)

def vector_to_str_list(x):
    return "\n".join([str(elem) for elem in x])


    

args = parse_args()
if args.interactive is not None:
    assert args.interactive in ['r2', 'non_model', 'free']
# Dynamically load network module.
net_module = imp.load_source('network', args.network)
gen_fn, disc_fn = getattr(net_module, 'get_network')()
# Dynamically load iterator module.
itr_module = imp.load_source('iterator', args.iterator)
itr_train, itr_val = getattr(itr_module, 'get_iterators')(args.batch_size)
itr_train_zipped = zip_iter(itr_train, itr_train)
itr_val_zipped = zip_iter(itr_val, itr_val)

if args.learn_m:
    gan_class = DepthNetGAN_M
elif args.sdam:
    gan_class = DepthNetGAN_SDAM
else:
    gan_class = DepthNetGAN
gan_kwargs = {
    'g_fn': gen_fn,
    'd_fn': disc_fn,
    'opt_d_args': {'lr':args.lr, 'betas':(args.beta1, args.beta2)},
    'opt_g_args': {'lr':args.lr, 'betas':(args.beta1, args.beta2)},
    'lamb': args.lamb,
    'detach': args.detach,
    'dnorm': args.dnorm,
    'sigma': args.sigma,
    'l2_decay': args.l2_decay,
    'use_l1': args.use_l1,
    'cheat': args.cheat,
    'no_gan': args.no_gan,
    'handlers': [save_handler("%s/%s" % (args.save_path, args.name))],
    'update_g_every': args.update_g_every,
    'use_cuda': False if args.cpu else 'detect'
}
net = gan_class(**gan_kwargs)
#net.alpha = args.alpha

if args.resume is not None:
    if args.resume == 'auto':
        # autoresume
        model_dir = "%s/%s/models" % (args.save_path, args.name)
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
if args.interactive is not None:
    if args.interactive == 'r2':
        # Basically compute the R2 over
        # the entire validation set.
        #process_data_one_sweep("tmp/test.csv")
        interactive.measure_pearson_one_sweep(net)
    elif args.interactive == 'non_model':
        # Evaluate the non-model on the valid set.
        net.eval_on_iterator(itr_val_zipped, use_gt_z=True)
    else:
        import pdb; pdb.set_trace()
        
else:
    net.train(
        itr_train=itr_train_zipped,
        itr_valid=itr_val_zipped,
        epochs=args.epochs,
        model_dir="%s/%s/models" % (args.save_path, args.name),
        result_dir="%s/%s" % (args.save_path, args.name),
        save_every=args.save_every
    )
