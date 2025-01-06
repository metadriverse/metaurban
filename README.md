# MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility

## 🛠 Quick Start
### Hardware Recommendation
We have tested the project on Linux, Windows WSL2 and MacOS. We strongly recommend using Ubuntu with Nvidia GPUs (with at least **8GB RAM** and **3GB VRAM**) to run the simulator. The performance benchmarks are conducted on different machines with **Nvidia RTX-3090, RTX-4080, RTX-4090, RTX-A5000 and Tesla V100**. It's normal that running PointNavigation env by `metaurban/examples/drive_in_static_env.py` with **~60FPS** and **~2GB** GPU Memory. 

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

install libs to use MetaUrban for RL training

```bash
pip install stable_baselines3 tensorboard wandb scikit-image pyyaml gdown
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
We provide docker file for MetaUrban. This works on machines with an NVIDIA GPU.To setup the MetaUrban using docker follow the below steps:
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
## 🤖 Run a Pre-Trained (PPO) Model 

We provide RL models trained on the task of navigation, which can be used to preview the performance of RL agents.

```bash
python -m metaurban.examples.drive_with_pretrained_policy
```

## 🚀 Model Training and Evaluation

We provide scripts for RL related research, based on **stable_baselines3**.

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

### Evaluation
We provide a script used to evaluate the quantitative performance of the RL agent
```bash
python RL/PointNav/eval_ppo.py --policy ./pretrained_policy_576k.zip
```
as an example of evaluting the provided policy.

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

## 📖 FAQs
For frequently asked questions about installing, RL training and other modules, please refer to: [FAQs](documentation/FAQs.md)
