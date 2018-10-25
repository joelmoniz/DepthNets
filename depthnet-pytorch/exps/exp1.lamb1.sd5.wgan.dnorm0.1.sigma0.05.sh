#!/bin/bash

source activate pytorch-env-py36
cd ..
NAME=exp1_lamb1_sd5_wgan_dnorm0.1_sigma0.05
python task_launcher_depthnet.py \
--name=${NAME} \
--batch_size=32 \
--epochs=100 \
--iterator=iterators/iterator.py \
--network=architectures/depthnet_shallowd5.py \
--save_every=5 \
--lamb=1.0 \
--cpu \
--dnorm=0.1 \
--sigma=0.05 \
--resume=auto
