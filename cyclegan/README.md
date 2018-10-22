# CycleGAN experiments for DepthNet

## Requirements

This code has been developed and tested on Python 3.6 and PyTorch 0.4.

To quickly get setup, we can create a new Conda environment and install the required packages, like so:

```
conda env create -f=environment.yml -n depthnet
```

## How to run

First download the .h5 files from here and put them the folder called `data/`:

* Face swapping: [faceswap.h5](https://mega.nz/#!9S5zFDYY!4KVJpbQXhAaOrVRFsyAuItyXbZyi7CW--1zmLffXF0Q) (md5hash 99604c2617b7f3743389f1604b7cad5f)
* Background synthesis: [depthnet_bg_vs_frontal.h5](https://mega.nz/#!RXpRFQSQ!xBKUuy0sxxKPacs8T6aCY9d2gkKnc2TJ5LUgB66NJ9g) (md5hash fbecb3c8aaa3f77ae71604f9a887730b)

### Cleaning up face swaps

One of the applications of CycleGAN was using it to clean up the operation which warps one face onto another. This involves using CycleGAN to learn a mapping between two domains: the domain of faces which have been pasted onto other faces (DepthNet faces) and the domain of ground truth faces. When this is trained, the mapping `depthnet -> real face` is the one we are interested in utilising.

![image](https://user-images.githubusercontent.com/2417792/46300240-34a4ec00-c571-11e8-8051-714e1a9baeca.png)

Some example images are shown below. (From left to right: source face, target face, DepthNet face, cleanup of DepthNet face)

![image](https://user-images.githubusercontent.com/2417792/46299419-4a191680-c56f-11e8-876c-f104950770ad.png) ![image](https://user-images.githubusercontent.com/2417792/46299425-4eddca80-c56f-11e8-8051-eb9e7d610273.png) ![image](https://user-images.githubusercontent.com/2417792/46299427-50a78e00-c56f-11e8-873c-85ec2a96ac58.png) ![image](https://user-images.githubusercontent.com/2417792/46299431-5309e800-c56f-11e8-8a3d-8a707fa30e67.png)
![image](https://user-images.githubusercontent.com/2417792/46299443-5604d880-c56f-11e8-8ed5-5abf6a21c33a.png) ![image](https://user-images.githubusercontent.com/2417792/46299448-58673280-c56f-11e8-9029-929f68cbdfef.png) ![image](https://user-images.githubusercontent.com/2417792/46299452-5ac98c80-c56f-11e8-821e-8c7df99e09bd.png) ![image](https://user-images.githubusercontent.com/2417792/46299454-5c935000-c56f-11e8-8097-a613b8af2bda.png)

To run this experiment, simply run:

```
python task_launcher_faceswap.py \
--name=experiment_faceswap \
--batch_size=16 \
--epochs=1000
```

### Background synthesis after face warp

Since DepthNet only warps the region corresponding to the face, it would be useful to be able to resynthesize the outside region such as the background and hair. In this experiment, CycleGAN maps from the domain consisting of DepthNet frontalised face and the background of the original face to the domain of ground truth (CelebA) images:

![image](https://user-images.githubusercontent.com/2417792/46300959-ff999900-c572-11e8-847f-bdf7fa5025ee.png)

Some examples are shown below. (From left to right: source image, source image + keypts, frontalised face with DepthNet, CycleGAN combining (3) and background of (1))

![image](https://user-images.githubusercontent.com/2417792/45967494-32381480-bffc-11e8-8002-d843ce926670.png)
![image](https://user-images.githubusercontent.com/2417792/45967500-349a6e80-bffc-11e8-9f07-bc4d9c2529a3.png)
![image](https://user-images.githubusercontent.com/2417792/45967504-36643200-bffc-11e8-8f61-6aebb649ffae.png)
![image](https://user-images.githubusercontent.com/2417792/45967506-382df580-bffc-11e8-8964-e1eb7ae30ace.png)


```
python task_launcher_bgsynth.py \
--name=experiment_depthnet_bg_vs_frontal \
--dataset=depthnet_bg_vs_frontal \
--batch_size=16 \
--network=architectures/block9_a6b3.py \
--epochs=500
```

You can find the pre-trained checkpoint for this [here](https://mega.nz/#!NL5EyYaL!tw_TS_F7zgOeRCxj3acBrazTPuvLDRx3igN1jA1Sdgg) (add `--resume=<path_to_pkl>` to the above script).

## Acknowledgements

* Some code has been used from the following repositories:
  * https://github.com/togheppi/CycleGAN
  * https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  * https://github.com/costapt/vess2ret
