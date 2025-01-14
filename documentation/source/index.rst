.. image:: figs/logo.png
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
   
   action_space.ipynb

.. raw:: html

    <table width="100%" style="margin: 0pt 0pt; text-align: center;">
        <tr>
            <td>
                <iframe style="display:block; width:100%; height:auto;"
                        src="https://www.youtube.com/watch?v=vHuAzNxmfKc&list=TLGGudtrAllTcWYxNDAxMjAyNQ&t=9s"
                        frameborder="0" 
                        allow="autoplay; encrypted-media; fullscreen; picture-in-picture"
                        allowfullscreen>
                </iframe>
            </td>
        </tr>
    </table>
    <br><br>

Relevant Projects
#################
.. raw:: html

    <b>
        MetaDrive: an Open-source Driving Simulator for AI and Autonomy Research
    </b> <br>
    Quanyi Li*, Zhenghao Peng*, Lan Feng, Qihang Zhang, Zhenghai Xue, Bolei Zhou
    <br>
    <i>IEEE TPAMI 2022</i><br>
    [<a href="https://arxiv.org/pdf/2109.12674.pdf" target="_blank">Paper</a>]
    [<a href="https://github.com/metadriverse/metadrive" target="_blank">Code</a>]
    [<a href="https://metadriverse.github.io/metadrive-simulator/" target="_blank">Webpage</a>]
    <br><br>

And more can be found in `MetaDriverse <https://metadriverse.github.io/>`_ .

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