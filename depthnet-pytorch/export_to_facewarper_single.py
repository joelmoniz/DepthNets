import numpy as np
import torch
from torch.autograd import (Variable,
                            grad)
import imp
import argparse
import os
import glob
from depthnet_gan import DepthNetGAN
from depthnet_gan_learnm import DepthNetGAN_M
from interactive import warp_to_target_face

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--src_kpts_file', type=str, required=True)
    parser.add_argument('--tgt_kpts_file', type=str, required=True)
    parser.add_argument('--src_img_file', type=str, required=True)
    parser.add_argument('--tgt_img_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_size', type=str, default='tgt',
                        help="Output size of the warp. Can be an integer, or 'tgt' or 'src'.")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--kpt_file_separator', type=str, default=',',
                        help="The separator used for the keypoint files. For example, if " +
                        "these are CSV then ',' should be used (which is the default value).")
    parser.add_argument('--learn_m', action='store_true')
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    return args    

args = parse_args()
# Dynamically load network module.
net_module = imp.load_source('network', args.network)
gen_fn, disc_fn = getattr(net_module, 'get_network')()

if args.learn_m:
    gan_class = DepthNetGAN_M
else:
    gan_class = DepthNetGAN
gan_kwargs = {
    'g_fn': gen_fn,
    'd_fn': disc_fn,
    'use_cuda': False if args.cpu else 'detect'
}
net = gan_class(**gan_kwargs)

if args.checkpoint is not None:
    print("Loading model: %s" % args.checkpoint)
    net.load(args.checkpoint)

if args.output_size not in ['src', 'tgt']:
    output_size = float(args.output_size)
else:
    output_size = args.output_size
    
warp_to_target_face(net,
                    src_kpts_file=args.src_kpts_file,
                    tgt_kpts_file=args.tgt_kpts_file,
                    src_img_file=args.src_img_file,
                    tgt_img_file=args.tgt_img_file,
                    output_dir=args.output_dir,
                    output_size=output_size,
                    kpt_file_separator=args.kpt_file_separator,
                    basename_prefix='tgt',
                    plot_warp=True)
