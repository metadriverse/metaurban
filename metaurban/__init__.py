from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame  # it is important to import pygame after that

from metaurban.envs import SidewalkStaticMetaUrbanEnv, SidewalkDynamicMetaUrbanEnv, TopDownMetaUrban, TopDownMetaUrbanEnvV2, TopDownSingleFrameMetaUrbanEnv
from metaurban.utils.registry import get_metaurban_class
import os

MetaUrban_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
