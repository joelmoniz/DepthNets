#!/bin/bash

cd ../..
source env.sh

# In this command, the target face is simply the source face. We generate rotations of the target face,
# first starting from the original position of the face to 0.785 radians (`tgt_angle_1`) (i.e. pi/4 radians,
# which is 90 degrees), and then from that position to -0.785 radians (`tgt_angle_2`). Note that since this
# rotation is being performed on the 3D target keypoints (before they are projected into 2D as input to DepthNet),
# we must already know the depth of the target. This means that the .csv file denoting the target keypoints must
# have a valid 3rd column, or else the script will raise an exception.

# Note that a more straightforward way to do the above is to simply estimate the depth of the source face and then
# rotate that instead. This means that a target face is only used to estimate the depth of the source. To do this,
# we can simply add the `--rotate_source` flag to the script. Note however that if this is used with the regular model
# (no GAN), the depths inferred are likely to be sub-par and therefore the rotation will also be sub-par. In that case,
# one can instead use the GAN model instead (see below).

# This is the no-GAN (regular) model
NAME=exp1_lamb1_sd5_nogan_sigma0_fixm
# This is the GAN model. Use this if you
# are using the --rotate_affine flag.
#NAME=exp1_lamb1_sd5_wgan_dnorm0.1_sigma0_fixm_repeat # gan model

mkdir input_facewarper

rm -r input_facewarper/rotation_example
python export_anim_to_facewarper.py \
--checkpoint=results/${NAME}/models/100.pkl \
--network=architectures/depthnet_shallowd5.py \
--axis=y \
--src_kpts_file=exps_warping/rotation_example/keypoints.txt \
--tgt_kpts_file=exps_warping/rotation_example/keypoints.txt \
--src_img_file=exps_warping/rotation_example/src.png \
--tgt_img_file=exps_warping/rotation_example/src.png \
--output_dir=input_facewarper/rotation_example \
--tgt_angle_1=0.785 \
--tgt_angle_2=-0.785 \
--scale_depth=1 \
--kpt_file_separator=' ' \
--rotate_source
