***links here***

# MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility

**`MetaUrban`** is a cutting-edge simulation platform designed for Embodied AI research in urban spaces. It offers:

- 🌆 **Infinite Urban Scene Generation**: Create diverse, interactive city environments.  
- 🏗️ **10,000 High-Quality Urban Objects**: Includes realistic infrastructure and clutter.  
- 🧍 **1,100 Human Models**: Each is rigged and equipped with 2,314 unique motions.
- 🤖 **11 Urban Agents**: Including delivery robots, cyclists, skateboarders, and more.  
- 🕹️ **Flexible User Interfaces**: Compatible with mouse, keyboard, joystick, and racing wheel.  
- 🎥 **Configurable Sensors**: Supports RGB, depth, semantic map, and LiDAR.  
- ⚙️ **Rigid-Body Physics**: Realistic mechanics for agents and environments.  
- 🌍 **OpenAI Gym Interface**: Seamless integration for AI and reinforcement learning tasks.

📖 Check out [**`MetaUrban` Documentation**](https://) to learn more!


***[A video here]***


## Latest Updates
- [25/01/25] **v0.1.0**: The first official release of MetaUrban :wrench: [[release notes]](https://github.com/StanfordVL/OmniGibson/releases/tag/v0.1.0)


## Table of Contents

- [MetaUrban](#metaurban-an-embodied-ai-simulation-platform-for-urban-micromobility)
  - [📎 Citation](#-citation)
  - [🛠 Quick Start](#-quick-start)
    - [Hardware Recommendations](#hardware-recommendations)
    - [Installation](#installation)
    - [Docker Setup](#docker-setup)
  - [🏃‍♂️ Simulation Environment Roam](#️-simulation-environment-roam)
    - [Point Navigation Environment](#point-navigation-environment)
    - [Social Navigation Environment](#social-navigation-environment)
  - [🤖 Run a Pre-Trained (PPO) Model](#-run-a-pre-trained-ppo-model)
  - [🚀 Model Training and Evaluation](#-model-training-and-evaluation)
    - [Reinforcement Learning](#reinforcement-learning)
      - [Training](#training)
      - [Evaluation](#evaluation)
  - [📖 Questions and Support](#questions-and-support)
  

## 📎 Citation

If you find MetaUrban helpful for your research, please cite the following BibTeX entry.

```latex
@article{wu2024metaurban,
  title={MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility},
  author={Wu, Wayne and He, Honglin and He, Jack and Wang, Yiran and Duan, Chenda and Liu, Zhizheng and Li, Quanyi and Zhou, Bolei},
  journal={arXiv preprint arXiv:2407.08725},
  year={2024}
}
```

## 🛠 Quick Start

### Hardware Recommendations

To ensure the best experience with **MetaUrban**, please review the following hardware guidelines:

- **Tested Platforms**:  
  - **Linux**: Supported and Recommended (preferably Ubuntu).
  - **Windows**: Works with **WSL2**.  
  - **MacOS**: Supported.  

- **Recommended Hardware**:  
  - **GPU**: Nvidia GPU with at least **8GB RAM** and **3GB VRAM**.  
  - **Storage**: Minimum of **10GB free space**.  

- **Performance Benchmarks**:  
  - Tested GPUs: **Nvidia RTX-3090, RTX-4080, RTX-4090, RTX-A5000, Tesla V100**.  
  - Example benchmark:  
    - Running `metaurban/examples/drive_in_static_env.py` achieves:  
      - ~**60 FPS**  
      - ~**2GB GPU memory usage**


### Installation

```bash
git clone -b dev https://github.com/metadriverse/metaurban
cd metaurban
conda create -n metaurban python=3.9
conda activate metaurban
pip install -e .
```

install ORCA algorithm for trajectory generation

```bash
conda install pybind11 -c conda-forge
cd metaurban/orca_algo && rm -rf build
bash compile.sh && cd ../..
```

it should be noticed that you should install ```cmake,make,gcc``` on your system before installing ORCA, more details can be found in [FAQs](documentation/FAQs.md).

install libs to use MetaUrban for RL training

```bash
pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown
```

assets will be downloaded automatically when first running the script 

`python metaurban/examples/drive_in_static_env.py`

if not, please download assets via the link:

"https://drive.google.com/file/d/1IL8FldCAn8GLa8QY1lryN33wrzbHHRVl/view?usp=sharing" for object assets

"https://drive.google.com/file/d/1XUGfG57Cof43dX2pkMYBhsFVirJ4DQ1o/view?usp=drive_link" for pedestrian assets

and organize the folder as:

```
-metaurban
  -metaurban
      -assets
      -assets_pedestrian
      -base_class
      -...
```

### Docker Setup
We provide a docker file for MetaUrban. This works on machines with an NVIDIA GPU. To set up the MetaUrban using docker, follow the below steps:
```bash
[sudo] docker -D build -t metaurban .
[sudo] docker run -it metaurban
cd metaurban/orca_algo && rm -rf build
bash compile.sh && cd ../.. 
```

then you can run the simulator in docker.

## 🏃‍♂️ Simulation Environment Roam
We provide examples to demonstrate features and basic usages of metaurban after the local installation.

### Point Navigation Environment

In a point navigation environment, there will be only static objects besides the ego agent in the scenario.

Run the following command to launch a simple scenario with manual control. Press `W,S,A,D` to control the delivery robot. 


```bash
python -m metaurban.examples.drive_in_static_env 
--density_obj 0.4
```

Press the key ```R``` to load a new scenario. If there is no response when you press `W,S,A,D`, press `T` to enable manual control.

### Social Navigation Environment
In a social navigation environment, there will be vehicles, pedestrians, and some other agents in the scenario.

Run the following command to launch a simple scenario with manual control. Press `W,S,A,D` to control the delivery robot. 

```bash
python -m metaurban.examples.drive_in_dynamic_env
--density_obj 0.4 --density_ped 1.0
```
## 🤖 Run a Pre-Trained (PPO) Model 

We provide RL models trained on the task of navigation, which can be used to preview the performance of RL agents.

```bash
python -m metaurban.examples.drive_with_pretrained_policy
```

## 🚀 Model Training and Evaluation

We provide scripts for RL-related research based on **stable_baselines3**.

### Reinforcement Learning
#### Training
```bash
python RL/PointNav/train_ppo.py
```
for PPO training in PointNav Env. You can change the parameters in the file.

```bash
python RL/SocialNav/train_ppo.py
```
for PPO training in Social Env. You can change the parameters in the file.

#### We only test RL training on Linux, there may be some issues on Windows and MacOS.

### Evaluation
We provide a script used to evaluate the quantitative performance of the RL agent
```bash
python RL/PointNav/eval_ppo.py --policy ./pretrained_policy_576k.zip
```
as an example of evaluating the provided policy.


## 📖 Questions and supports

For frequently asked questions about installing, RL training and other modules, please refer to: [FAQs](documentation/FAQs.md)

Can't find the answer to your question? Try asking the developers and community on our Discussions forum.

***[open a discussion forum in GitHub]***