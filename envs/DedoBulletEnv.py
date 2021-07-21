#wrapper for DEDO environments
#these are necessary because garage do not call gym.make with kwargs

from gym.wrappers import TimeLimit

import dedo
from dedo.utils.args import get_args as dedo_get_args
from dedo.envs.deform_env import DeformEnv

class HangBagBulletEnv(DeformEnv):

    def __init__(self, render=False):
        dedo_args = dedo_get_args()[0]
        dedo_args.task = 'HangBag'
        dedo_args.viz = render
        version = 0
        super().__init__(version=version, args=dedo_args)