import numpy as np
import torch
from torch.autograd import (Variable,
                            grad)
import imp
import argparse
import os
from depthnet_gan import DepthNetGAN
from depthnet_gan_learnm import DepthNetGAN_M
from interactive import warp_to_rotated_target_face

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--src_kpts_file', type=str, required=True,
                        help="Source keypoints file.")
    parser.add_argument('--tgt_kpts_file', type=str, required=True,
                        help="Target keypoints file.")
    parser.add_argument('--src_img_file', type=str, required=True,
                        help="Source image file.")
    parser.add_argument('--tgt_img_file', type=str, required=True,
                        help="Target image file.")
    parser.add_argument('--axis', choices=['x', 'y', 'z'],
                        default='x',
                        help="The axis around which we perform the rotation.")
    parser.add_argument('--tgt_angle_1', type=float, default=np.pi / 2.,
                        help="The first angle (in rads) to rotate the face toward.")
    parser.add_argument('--tgt_angle_2', type=float, default=0.,
                        help="The second angle (in rads) to rotate the face toward.")
    parser.add_argument('--scale_depth', type=float, default=100.,
                        help="Scale the depth by this constant. If this is == 0, " +
                        "then the depths are scaled so that they are in range [0,1].")
    parser.add_argument('--rotate_source', action='store_true',
                        help="If set, we do not warp the source face to target. Instead, " +
                        "we rotate the source directly (after estimating its depth based " +
                        "on a target face), and manually construct affines to perform the rotation.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory")
    parser.add_argument('--output_size', type=str, default='tgt',
                        help="Output size of the warp. Can be an integer, or 'tgt' or 'src'.")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="x/y target coordinates are shifted by this amount prior to rotation. " +
                        "After the rotation, these coords are de-shifted with this mean.")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="x/y target coordinates are scaled by this amount prior to rotation. " +
                        "After the rotation, these coordinates are de-scaled with this std.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Model checkpoint .pkl")
    parser.add_argument('--learn_m', action='store_true',
                        help="This flag must be set if you're using the DepthNet " +
                        "model which explicitly predicts the affine parameters.")
    parser.add_argument('--kpt_file_separator', type=str, default=',',
                        help="The separator used for the keypoint files. For example, if " +
                        "these are CSV then ',' should be used (which is the default value).")
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
    
warp_to_rotated_target_face(
    net=net,
    src_kpts_file=args.src_kpts_file,
    tgt_kpts_file=args.tgt_kpts_file,
    axis=args.axis,
    tgt_angle_1=args.tgt_angle_1,
    tgt_angle_2=args.tgt_angle_2,
    src_img_file=args.src_img_file,
    tgt_img_file=args.tgt_img_file,
    output_dir=args.output_dir,
    output_size=output_size,
    norm_mean=args.norm_mean,
    norm_std=args.norm_std,
    scale_depth=args.scale_depth,
    rotate_source=args.rotate_source,
    kpt_file_separator=args.kpt_file_separator
)
