.. _getting_start:

#############################
Getting Start with MetaUrban
#############################

Tryout MetaUrban with one line
###############################

We provide a script to let you try out MetaUrban by keyboard immediately after installation! Please run::

    # Make sure current folder does not have a sub-folder named metaurban
    python -m metaurban.examples.drive_in_static_env

Press `T` in the main window will kick-off this.
You can also press `H` to visit the helper information on other shortcuts.

We also provide a script to let you experience an "auto-drive" journey carried out by our pre-trained RL agent. Please run::

    python -m metaurban.examples.drive_with_pretrained_policy

Besides, you can verify the efficiency of MetaUrban via running::

    python -m metaurban.tests.test_env.profile_metaurban

MetaUrban provides 2 sets of RL environments: the static environments and the dynamic environments.
We provide the examples for those suites as follow:

.. code-block::

    # Make sure current folder does not have a sub-folder named metaurban

    # ===== Environments with only Static Objects =====
    python -m metaurban.examples.drive_in_static_env

    # ===== Environments with Static Objects, Vehicles, Pedestrians and Robots =====
    python -m metaurban.examples.drive_in_dynamic_env

Using MetaUrban in Your Code
#############################

The usage of MetaUrban is as same as other **gym** environments.
Almost all decision making algorithms are compatible with MetaUrban, as long as they are compatible with OpenAI gym.
The following scripts is a minimal example for instantiating a MetaUrban environment instance.

.. code-block:: python

    from metaurban.envs import SidewalkStaticMetaUrbanEnv
    import gymnasium as gym

    env = SidewalkStaticMetaUrbanEnv(dict(use_render=True, num_scenarios=1000, start_seed=1010, object_density=0.05))
    obs, info = env.reset()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()

.. Note:: Please note that each process should only have one single MetaUrban instance due to the limit of the underlying simulation engine.
    Thus the parallelization of training environment should be in process-level instead of thread-level.
