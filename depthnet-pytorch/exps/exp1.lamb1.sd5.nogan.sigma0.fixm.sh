#!/bin/bash

cd ..
source env.sh
NAME=exp1_lamb1_sd5_nogan_sigma0_fixm
python task_launcher_depthnet.py \
--name=${NAME} \
--batch_size=32 \
--epochs=100 \
--iterator=iterators/iterator.py \
--network=architectures/depthnet_shallowd5.py \
--save_every=10 \
--lamb=1.0 \
--cpu \
--no_gan \
--resume=auto
