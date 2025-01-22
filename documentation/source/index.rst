.. image:: figs/logo_white.png
   :width: 1800
   :align: center


|
|

########################
MetaUrban Documentation
########################


Welcome to the MetaUrban documentation!
MetaUrban is a compositional simulation platform for AI-driven urban micromobility research with the following key features: 

* Compositional: It supports generating infinite scenes with various road maps and traffic settings for the research of generalizable RL.
* Lightweight: It is easy to install and run.
* Realistic: Accurate physics simulation and multiple sensory input including Lidar, RGB images, top-down semantic map and first-person view images.

This documentation brings you the information on installation, usages and more of MetaUrban!

You can also visit the `GitHub repo <https://github.com/metadriverse/metaurban>`_ of MetaUrban.
Please feel free to contact us if you have any suggestions or ideas!


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   before_reading.ipynb
   install.rst
   get_start.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: RL Training with MetaUrban

   rl_environments.ipynb
   config_system.ipynb
   observation.ipynb
   action.ipynb
   reward_cost_done.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer's Guide

   new_env.ipynb
   system_design.ipynb
   debug_mode.ipynb
   sensors.ipynb
   top_down_render.ipynb
   points_and_lines.ipynb
   log_msg.ipynb
   ros.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Applications

.. raw:: html

    <div style="margin: 0pt 0pt; text-align: center;">
    <iframe width="800" height="480" src="https://www.youtube.com/embed/vHuAzNxmfKc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>

Citation
########

You can read `our white paper <https://arxiv.org/pdf/2407.08725.pdf>`_ describing the details of MetaUrban! If you use MetaUrban in your own work, please cite:

.. code-block:: latex

    @article{wu2024metaurban,
             title={MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility},
             author={Wu, Wayne and He, Honglin and He, Jack and Wang, Yiran and Duan, Chenda and Liu, Zhizheng and Li, Quanyi and Zhou, Bolei},
             journal={arXiv preprint arXiv:2407.08725},
             year={2024}
   }