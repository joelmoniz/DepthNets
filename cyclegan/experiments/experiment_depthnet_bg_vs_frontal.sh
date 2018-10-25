#!/bin/bash

# cg_baseline_sina_newblock9_inorm_sinazoom

cd ..

python task_launcher_bgsynth.py \
--name=experiment_depthnet_bg_vs_frontal \
--dataset=depthnet_bg_vs_frontal \
--batch_size=16 \
--network=architectures/block9_a6b3.py \
--epochs=1000
