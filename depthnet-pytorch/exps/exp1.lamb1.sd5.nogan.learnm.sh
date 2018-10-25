#!/bin/bash

source activate pytorch-env-py36
env
cd ..
NAME=exp1_lamb1_sd5_nogan_learnm
python task_launcher_depthnet.py \
--name=${NAME} \
--batch_size=32 \
--epochs=100 \
--iterator=iterators/iterator.py \
--network=architectures/depthnet_sd5_learnm.py \
--save_every=20 \
--lamb=1.0 \
--cpu \
--no_gan \
--learn_m \
--resume=auto \
