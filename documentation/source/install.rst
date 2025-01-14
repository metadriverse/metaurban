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
    conda install pybind11 -c conda-forge
    cd metaurban/orca_algo && rm -rf build
    bash compile.sh && cd ../..

    # install the requirements for reinforcement learning, imitation learning and visualization
    pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown

.. note:: Using ``git clone https://github.com/metadriverse/metaurban.git --single-branch``
  will only pull the main branch and bypass other branches, saving disk space.

.. note:: ``cmake, make and gcc`` are required to compile the ORCA module, please install them first.

Pull assets
############################################
After having the source code installed, 3D assets are still required to run MetaUrban.
Generally, they will be pulled automatically when you run any MetaUrban program for the first time.
But you can still pull the asset manually by::

 python -m metaurban.pull_asset

.. note:: All ``python -m`` scripts are supposed to be runnable in all places **except** in the working direction that has a sub-folder called :code:`./metaurban`.

Verify installation
#############################
To check whether MetaUrban is successfully installed, please run the following code::

    python -m metaurban.tests.test_env.profile_metaurban

This script can also verify the efficiency of MetaUrban through the printed messages.
The default observation contains information on ego vehicle's states, Lidar-like cloud points showing neighboring vehicles, and information about navigation and tasks. 
If the above observation is not enough for your RL algorithms and you wish to use the Panda3D camera to provide realistic RGB images as the observation, please continue reading this section.

Install MetaUrban with headless rendering
#############################


Install MetaUrban with advanced offscreen rendering
#############################

Known Issues
######################
.. note:: Run MetaUrban on a machine without monitor / X-server
