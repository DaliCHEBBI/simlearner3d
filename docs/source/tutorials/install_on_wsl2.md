# Install Simlearner3d on WSL2 with CUDA support

## [Setting up WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)

Simlearner3D requires the latest Linux distros to work properly. That's why it's important to make sure that everything in the WSL is up to date.

1. You must be running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11 to use the commands below. If you are on earlier versions please see the [manual install page](https://learn.microsoft.com/en-us/windows/wsl/install-manual).
2. Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator", enter the ``wsl --install`` command, then restart your machine.
3. Ensure you have the latest WSL kernel:
        
        wsl.exe --update
4. This command will enable the features necessary to run WSL and install the Ubuntu distribution (default Ubuntu 22.04) of Linux.
5. If you run ``wsl --install`` and see the WSL help text, that means WSL is already installed. In that case run ``wsl --list --online`` to see a list of available distros and run ``wsl --install -d <DistroName>`` to install a distro. In this case we need Ubuntu 22.04, so type:

        wsl --install -d Ubuntu
6. You need to update your libraries before moving forward by running 

        sudo apt update && sudo apt upgrade

## [Installing Anaconda](https://docs.anaconda.com/anaconda/install/linux/)

1. In your browser, download the Anaconda installer for [Linux](https://www.anaconda.com/products/distribution#linux).
2. It's recommended to copy the downloaded file to your WSL home directory- ``\\wsl.localhost\Ubuntu\home\<username>``
3. In the Ubuntu/ WSL terminal, run the following-

        bash Anaconda3-2022.10-Linux-x86_64.sh
    Replace the ``Anaconda3-2022.10-Linux-x86_64.sh`` part with your downloaded file name.
4. Press Enter to review the license agreement. Then press and hold Enter to scroll.
5. Enter “yes” to agree to the license agreement.
6. Use Enter to accept the default install location, the installer displays PREFIX=/home/<USER>/anaconda<2/3> and continues the installation. It may take a few minutes to complete.
7. The installer prompts you to choose whether to initialize Anaconda Distribution by running conda init. Anaconda recommends entering “yes”.

## Installing Simlearner3D

1. Clone the Simlearner3D master branch in your working directory using the following command
        
        
        git clone https://github.com/DaliCHEBBI/simlearner3d.git
       
   After it's done, navigate to the `environment.yml` file. You might need to make couple of changes here before installing the libararies.
        
3. We use anaconda to manage virtual environments. This makes installing pytorch-related libraries way easier than using pure pip installs.

4. To install the virtual environment, run the following commands within Ubuntu/ WSL terminal-

        conda env create -f "/PATH/TO/environment.yml"

    It will take couple of minutes to download and install all the packages. After that activate the environment by running-

        conda activate simlearner3d

5. Then install from a specific branch from github directly.

        pip install --upgrade https://github.com/DaliCHEBBI/simlearner3d/tarball/master
        
   Alternatively, navigate your working directory to the cloned simlearner3d directory and install from sources directly in editable mode with

        pip install -e .

At this point Simlearner3D is installed and you can move ahead with inference or testing using the method stated [here](https://ignf.github.io/simlearner3d/tutorials/make_predictions.html) 


        
## Troubleshooting

- *ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found* ([**](https://askubuntu.com/a/582910))

    - run the following commands in your Ubuntu/ WSL terminal

            sudo add-apt-repository ppa:ubuntu-toolchain-r/test
            sudo apt-get update
            sudo apt-get install --only-upgrade libstdc++6
- GPU-related errors: 
        
    - *RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx*
    - *Failed to initialize NVML: GPU access blocked by the operating system Failed to properly shut down NVML: GPU access blocked by the operating system*
    - *Failed to initialize NVML: Driver/library version mismatch*
    - any other GPU related errors

        - Make sure you followed the cuda installation part as well as the cuda toolkit version matching properly. If not, remove cuda completely and install again. [**](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)
        - Make sure to open the command prompt in Admin mode [**](https://forums.developer.nvidia.com/t/failed-to-properly-shut-down-nvml-gpu-access-blocked-by-the-operating-system/234413/5)
        - In some cases using the Admin mode blocks access to the GPU, use non-elevated command prompt if that occurs
        - Restart your WSL and try again [**](https://stackoverflow.com/a/43023000/8889660)
                
                wsl --shutdown

        - If the error persists then see if you have the correct GPU models. As of now, cuda toolkit in WSL is supported in **NVIDIA GeForce Game Ready or NVIDIA RTX/Quadro card**s only.


        

