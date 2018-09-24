# CycleGAN experiments for DepthNet

## Requirements

This code has been developed and tested on Python 3.6 and PyTorch 0.4.

To quickly get setup, we can create a new Conda environment and install the required packages, like so:

```
conda create -n depthnet python=3.6
conda install pytorch torchvision -c pytorch
pip install tqdm
```

## How to run

First download the .h5 files from here and put them the folder called `data/`:

* Face swapping: [faceswap.h5](https://mega.nz/#!9S5zFDYY!4KVJpbQXhAaOrVRFsyAuItyXbZyi7CW--1zmLffXF0Q) (md5hash 99604c2617b7f3743389f1604b7cad5f)
* Background synthesis: [depthnet_bg_vs_frontal.h5](https://mega.nz/#!RXpRFQSQ!xBKUuy0sxxKPacs8T6aCY9d2gkKnc2TJ5LUgB66NJ9g) (md5hash fbecb3c8aaa3f77ae71604f9a887730b)

### Cleaning up face swaps

One of the applications of CycleGAN was using it to clean up the operation which warps one face onto another. This involves blending the pasted face into the face that it was pasted onto, e.g. from:

<!-- ![image](https://user-images.githubusercontent.com/2417792/45967635-9d81e680-bffc-11e8-8317-c1f83bae4c3e.png)
![image](https://user-images.githubusercontent.com/2417792/45967641-9fe44080-bffc-11e8-911e-c17695573fed.png)
![image](https://user-images.githubusercontent.com/2417792/45967644-a1ae0400-bffc-11e8-8048-00fb69d389f2.png)
![image](https://user-images.githubusercontent.com/2417792/45967646-a4105e00-bffc-11e8-9912-22b482b2c0f8.png) -->

![img1](https://user-images.githubusercontent.com/2417792/45315224-d05ab380-b501-11e8-9ec5-98aeb770a9e4.png)

to:

![img2](https://user-images.githubusercontent.com/2417792/45315235-d81a5800-b501-11e8-9fd8-3f81126d518f.png)

This means that CycleGAN tries to map between two domains: the domain of pasted faces (first image) and the domain of (ground truth) CelebA images.

To run this experiment, simply run:

```
python task_launcher_faceswap.py --name=my_experiment_name
```

### Background synthesis after face warp

Since DepthNet only warps the region corresponding to the face, it would be useful to be able to resynthesize the outside region such as the background and hair.

![image](https://user-images.githubusercontent.com/2417792/45967494-32381480-bffc-11e8-8002-d843ce926670.png)
![image](https://user-images.githubusercontent.com/2417792/45967500-349a6e80-bffc-11e8-9f07-bc4d9c2529a3.png)
![image](https://user-images.githubusercontent.com/2417792/45967504-36643200-bffc-11e8-8f61-6aebb649ffae.png)
![image](https://user-images.githubusercontent.com/2417792/45967506-382df580-bffc-11e8-8964-e1eb7ae30ace.png)

From left to right:
* (1) source image
* (2) source image + keypts
* (3) warped face with DepthNet
* (4) CycleGAN combining (3) and background of (1)

In this experiment, CycleGAN maps from the domain consisting of DepthNet warped face (col 3) and the background of the original face (col 1) to the domain of ground truth (CelebA) images:

```
python task_launcher_bgsynth.py \
--name=experiment_depthnet_bg_vs_frontal \
--dataset=depthnet_bg_vs_frontal \
--batch_size=16 \
--network=architectures/block9_a6b3.py \
--epochs=1000
```

## Acknowledgements

* Some code has been used from the following repositories:
  * https://github.com/togheppi/CycleGAN
  * https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  * https://github.com/costapt/vess2ret
