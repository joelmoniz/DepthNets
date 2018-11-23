#!/bin/bash

cd ../..
source env.sh

NAME=exp1_lamb1_sd5_nogan_sigma0_fixm
#NAME=exp1_lamb1_sd5_wgan_dnorm0.1_sigma0_fixm

mkdir input_facewarper

OUT_FOLDER=warp_example

echo "removing old folder..."
cd facewarper_input
rm -r ${OUT_FOLDER}
cd ..
echo "running facewarper..."
python export_to_facewarper_single.py \
--network=architectures/depthnet_shallowd5.py \
--checkpoint=results/${NAME}/models/100.pkl \
--src_kpts_file=exps_warping/warp_example/src_keypoints.txt \
--tgt_kpts_file=exps_warping/warp_example/obama_keypoints.txt \
--src_img_file=exps_warping/warp_example/src.png \
--tgt_img_file=exps_warping/warp_example/obama.png \
--output_dir=facewarper_input/${OUT_FOLDER} \
--kpt_file_separator=' '
