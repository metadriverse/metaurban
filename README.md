# MetaUrban: A Simulation Platform for Embodied AI in Urban Spaces


## 🛠 Quick Start
Install MetaUrban (Tiny Version) via:

```bash
git clone -b tiny https://github.com/metadriverse/metaurban
cd metaurban
pip install -e .
```

download assets from

```
https://drive.google.com/file/d/194pgea_J7mjjlmFD4pzj3KQAWxsJtmeL/view?usp=sharing
```

unzip the file and organize the folder as

```
-metaurban
  -metaurban
      -assets
      -assets_pedestrian
      -base_class
      -...
```

install ORCA algorithm for trajectory generation

```bash
conda install pybind11 -c conda-forge
pip install scikit-image
cd metaurban/orca_algo
rm -rf build
bash compile.sh 
```

install torch and stable-baselines3 for RL training

```bash
pip install torch
pip install stable_baselines3
```

we put a small asset subset of objects and agents under the folder `metaurban/assets/models/test/` and `metaurban/assets_pedestrian/`. You should change paths in `path_config.yaml`.

```bash
metaurbanasset: /PATH/TO/ASSETS

parentfolder: /PATH/TO/AdjustedParameters
```

Note that the program is tested on Linux, Windows and WSL2. Some issues in MacOS wait to be solved.


## MetaUrban Simulator Roam
We provide examples to demonstrate features and basic usages of metaurban after the local installation.

### Point Navigation Environment

In point navigation environment, there will be only static objects besides the ego agent in the scenario.

Run the following command to launch a simple scenario with manual control. Press `W,S,A,D` to control the delivery robot. 


```bash
python -m metaurban.examples.drive_in_static_env
```

Press key ```R``` for loading a new scenario. If there is no response when you press `W,S,A,D`, press `T` to enable manual control.

### Social Navigation Environment
In social navigation environment, there will be vehicles, pedestrians and some other agents in the scenario.

Run the following command to launch a simple scenario with manual control. Press `W,S,A,D` to control the delivery robot. 

```bash
python -m metaurban.examples.drive_in_dynamic_env
```

## MetaUrban-12K Dataset Generation
We provide a subset of seeds with selected ORCA reference trajectory for ego agent. You can run the command as below to generate a scenario from these seeds

```bash
python -m metaurban.scripts.generate_static_scenario
```

for PointNav environment 

```bash
python -m metaurban.scripts.generate_dynamic_scenario
```

for SocialNav environment 

#### The file `valid_seed.pkl` in the root folder is a subset of `MetaUrban-12K` Dataset. It's notable that currently provided version do not match the version we use since complete asset library is not provided yet, there may still be some unreasonable reference trajectories in the scenario.
