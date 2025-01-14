try:
    import torch

    assert hasattr(torch, "device")
    from metaurban.examples.ppo_expert.torch_expert import torch_expert as expert
except:
    from metaurban.examples.ppo_expert.numpy_expert import expert
