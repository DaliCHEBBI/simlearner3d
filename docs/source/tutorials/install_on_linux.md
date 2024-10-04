# Install Simlearner3D on Linux

## Setting up a virtual environment

### Prerequisites

We use [anaconda](https://www.anaconda.com/products/individual) to manage virtual environments.
This makes installing pytorch-related libraries way easier than using pure pip installs.

We enable CUDA-acceleration in pytorch as part of the defaut virtual environment recipe (see below).

### Environment Installation

To install the environment, follow these instructions:

```bash
# Install mamba to create the environment faster
conda install -y mamba -n base -c conda-forge
# Build it with mamba
mamba env create -f environment.yml
# activate it
conda activate simlearner3d
```

> CUDA: if you have CUDA capable devices, [check you CUDA version](https://varhowto.com/check-cuda-version/), and check that the `cudatoolkit` version in `setup_env/requirements.yml` matches yours or at least is compatible with your installed version.
Using an older cuda-toolkit will probably work on a newer system thanks to NVIDIA backward compatibility (check the [compatibility matrix]([url](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package))). For instance: installing pytorch wheels for Cuda 11.3 (`cu113`) will still work on a system with NVIDIA 12.1 installed if the NVIDIA drivers are recent enough.


Finally, activate the created environment by running

```bash
conda activate simlearner3d
```

## Install source as a package

If you are interested in running inference from anywhere, the easiest way is to install code as a package in a your virtual environment.

Start by activating the virtual environment with

```bash
conda activate simlearner3d
```

Then install the latest version from pypi.
**Warning:** activating the environment is required as the public pip package does not
handle its dependencies!
```
pip install simlearner3d
```

Or install from a specific branch from github directly. Argument `branch_name` is "master" for now.
```
pip install --upgrade https://github.com/DaliCHEBBI/simlearner3d/tarball/{branch_name}
```

Alternatively, you can install from sources directly in editable mode with
```bash
pip install --editable .

```


## Troubleshooting
