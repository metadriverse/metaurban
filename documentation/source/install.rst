.. _install:

######################
Installing MetaUrban
######################


Install MetaUrban
############################################

The installation of MetaUrban on different platforms is straightforward and easy!
We recommend to use the command following to install::

    git clone https://github.com/metadriverse/metaurban.git
    cd metaurban
    
    # install metaurban
    conda create -n metaurban python=3.9
    conda activate metaurban
    pip install -e .

    # install module ORCA for trajectory generation of pedestrians in urban environments
    # it should be noticed that you should install cmake, make, gcc on your system before installing ORCA
    conda install pybind11 -c conda-forge
    cd metaurban/orca_algo && rm -rf build
    bash compile.sh && cd ../..

    # install the requirements for reinforcement learning, imitation learning and visualization
    pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown

