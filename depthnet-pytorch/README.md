# DepthNets (in PyTorch)

DepthNet estimating both depth and affine params:
<p align="center">
  <img src="figures/DepthNet_diagram_with_Kpt.jpg" width="600"/>
</p>

DepthNet estimating only depth:
<p align="center">
  <img src="figures/DepthNet_diagram_only_depth.jpg" width="600"/>
</p>


We train DepthNet on unpaired faces belonging to different identities and compare with other models that estimate depth.
We use the 3DFAW dataset that contains 66 3D keypoints to facilitate comparing with ground truth (GT) depth.  It provides
13,671 train and 4,500 valid images. We extract from the valid set, 75 frontal, left and right looking faces yielding a total
of 225 test images, which provides a total of 50,400 source and target pairs.  We train the psuedoinverse DepthNet model that
relies on only keypoints.

Concretely, this is the loss function we wish to minimize:

<!--  % RENDER WITH LATEXIT
\Bigg\|
\boldsymbol{x}_{t} -
%
\underbrace{\begin{bmatrix} m_1 & m_2 & m_3 & t_x \\ m_4 & m_5 & m_6 & t_y \end{bmatrix}}_{\boldsymbol{m}}
%
\left[ \begin{array}{c} \boldsymbol{x}_s \\ g(\boldsymbol{x}_s, \boldsymbol{x}_t) \\ 1 \end{array} \right] \Bigg\|^2
-->

<img src="https://user-images.githubusercontent.com/2417792/46366635-96d12000-c649-11e8-83af-dfedf4b57dd7.png" width=500 />

where the LHS is a `(2, k)` matrix (where `k` denotes the number of keypoints), `m` is a `(2, 4)` matrix, and the RHS is a `(4, k)` matrix. The DepthNet model is `g(x_s, x_t)`, which takes both the source and target keypoints and tries to estimate the depth of the source keypoints. As mentioned earlier, `m` can be found as a closed form solution, which is the pseudoinverse DepthNet model. However, one can also use the network `g` to also predict `m`, and we also run this experiment (more on that later).

We also train a variant of DepthNet that applies an adversarial loss on the depth values (DepthNet+GAN).
This model uses a conditional discriminator that is conditioned on 2D keypoints and discriminates GT from estimated depth values. 
The model is trained with both keypoint and adversarial losses.

## Requirements

This code has been developed and tested on Python 3.6 and PyTorch 0.4.

To quickly get setup, we can create a new Conda environment and install the required packages, like so:

```
conda env create -f=environment.yml -n depthnet
```

## Training

### Data

You will need to obtain the [3DFAW data](http://mhug.disi.unitn.it/workshop/3dfaw/) for this. You can do this by filling a data request form and sending it to the organisers of the data. When this is done, extract the zip files in some directory (provided by the organisers) so that the folders `train_lm`, `valid_lm`, `train_img`, and `valid_img` exist. Also download the [valid/test split file](https://mega.nz/#!FD5HBa7a!AZoP_TmvWaDsN5YV0coVMHU9fL166wgHoBFw5ixgdBU) and place it in the same directory.

Then, `cp env.sh.example env.sh`, modify `env.sh` to point to this 3DFAW directory, then `source env.sh`. Afterwards, run `prepare_dataset.py`, which will generate some `.npz` files.

### Experiments

* (1) `exps/exp1.lamb1.sd5.nogan.sigma0.05.sh`: this is the baseline experiment. This corresponds to the DepthNet pseudoinverse model.
* (2) `exps/exp1.lamb1.sd5.nogan.learnm.sh`: the DepthNet model where `g()` also learns the `m`.
* (3) `exps/exp1.lamb1.sd5.wgan.dnorm0.1.sigma0.05.sh`: (1) but GANified, with a conditional descriminator on the predicted depths.
* (4) `exp1.lamb1.sd5.wgan.dnorm0.1.learnm.sh`: (3) but with learning `m`.

Once trained, the results and diagnostic files will be located in `results/<experiment_name>`. Models can be reloaded with the `--resume=<path_to_checkpoint>` flag, but since this is set to `--resume=auto` in the script, whenever the experiment is run it will try to find the latest model checkpoint and load that instead. Pre-trained model checkpoints can be found [here](https://mega.nz/#F!FHoT0KIb!09aEueFerQ0zzuJvvN5FnA).

## Evaluation

Once a model has been trained, add the `--interactive` flag to the experiment script. Instead of training the model, this will put you in a PDB debug mode. From this, one can invoke various functions to compute useful statistics based on the model, such as those shown in the paper:

* Depth correlation (called 'DepthCorr' in the paper). This computes the cross-correlation matrix between the X = inferred depths and Y = the ground truth ones and computes the trace of the matrix (the higher the trace, the better). Because DepthNet requires a source and target face and that both these faces can be in one of three orientations (left-facing, center-facing, or right-facing), we compute a 3x3 matrix of traces `M` instead, where `M[i,j]` is the trace of the correlation matrix between: X (inferred depths when mapping to orientation `j` faces using orientation `i` faces), ground truth depths Y.
  * `interactive.measure_depth_test_pairwise(net, grid=True)`
  * There is also a `dump_file` flag if you want to save the predicted depths to disk as an .npz file.
* Squared error between the target and predicted target keypoints (i.e. the above equation)
  * `interactive.measure_kp_error_test_pairwise(net, grid=False)`
  
## Reproducing figures

TODO
