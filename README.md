<div align="center">

# Simlearner3D : Learning similarity for 3D reconstruction from aerial and satellite imagery


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)

[![CICD](https://github.com/DaliCHEBBI/simlearner3d/actions/workflows/cicd.yaml/badge.svg)](https://github.com/DaliCHEBBI/simlearner3d/actions/workflows/cicd.yaml)
[![Documentation Build](https://github.com/DaliCHEBBI/simlearner3d/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/DaliCHEBBI/simlearner3d/actions/workflows/gh-pages.yml)
</div>
<br><br>


Simlearner3D is a deep learning library that allows to learn similarity cues for large scale 3D reconstruction from aerial and satellite imagery. 

It is built to prepare, structure training/evaluation/test datasets, train a variety of neural networks and test the performance of learned models.

Simlearner3d is built upon [PyTorch](https://pytorch.org/).

Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
which heavily relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to enable flexible and rapid iterations of deep learning experiments.


> &rarr; For installation and usage, please refer to [**Documentation**](https://dalichebbi.github.io/simlearner3d/).


Please cite simlearner3d if it helped your own research. Here is an example BibTex entry:
```
@INPROCEEDINGS{10208916,
author={Chebbi, Mohamed Ali and Rupnik, Ewelina and Pierrot-Deseilligny, Marc and Lopes, Paul},
booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
title={DeepSim-Nets: Deep Similarity Networks for Stereo Image Matching}, 
year={2023},
pages={2097-2105}}

```