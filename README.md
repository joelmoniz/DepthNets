# Unsupervised Depth Estimation, 3D Face Rotation and Replacement (NIPS 2018)

This repository is split up into multiple sections, which each address different tasks in the paper. Click on the corresponding headers to be taken to those sections in this repository. Each section has a README.md file detailing how to set everything up.

If you are having issues with running the code, please do not hesitate to submit an issue.

## [DepthNets](depthnet-pytorch)

<img src="depthnet-pytorch/figures/DepthNet_only_kpts.jpg" width=500 />

This contains code to run DepthNet, and in particular section 3.2 ("Evaluation on unpaired faces and comparison to other models").

## [FaceWarper](FaceWarper)

<img src="https://user-images.githubusercontent.com/627828/47393012-5450af80-d6ec-11e8-9fdf-58b37eb8749a.png" width=500 />

This is the OpenGL pipeline used to produce face warps based on the depths and geometry output by DepthNet.

## [CycleGAN](cyclegan)

This contains code to run the experiments detailed in section 3.3 ("Face rotation, replacement, and adversarial repair"). Note that some of the data preparation code here is dependent on the compilation of FaceWarper.

<img src="https://user-images.githubusercontent.com/2417792/46300959-ff999900-c572-11e8-847f-bdf7fa5025ee.png" width=500 />
<img src="https://user-images.githubusercontent.com/2417792/46300240-34a4ec00-c571-11e8-8051-714e1a9baeca.png" width=500 /> 

## Disclaimer

We have identified that the work we have presented has the potential to be applied in a manner which could be controversial (for example, see [Deepfakes](https://en.wikipedia.org/wiki/Deepfake)). We do not in any manner endorse the use of this code in malicious or deceptful applications. Please use this code responsibly!
