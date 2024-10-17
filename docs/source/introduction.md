Simlearner3D is a deep learning library designed with the focus of large scale dense image matching by similarity learning from pairs of epipolar images.

The library implements the training of feature extractors (MSAFF,UNet32,UNet-Attention) with and without a MLP given pairs of images, corresponding disparity and occlusion maps. Qualification of models is conducted using joint probability maps estimation. This actually tells if learned similarity cues allow decent separation between matching and non matching pixels.


Simlearner3D is built upon [PyTorch](https://pytorch.org/). It keeps the standard data format 
from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/). 
Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
The latter relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to 
enable flexible and rapid iterations of deep learning experiments.