#!/bin/bash

# cg_baseline_sina_newblock9_inorm_sinazoom_faceswap

cd ..

python task_launcher_faceswap.py \
--name=experiment_faceswap \
--batch_size=16 \
--epochs=1000
