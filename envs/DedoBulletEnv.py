#wrapper for DEDO environments
#these are necessary because garage do not call gym.make with kwargs
import numpy as np
import gym

from dedo.utils.mesh_utils import get_mesh_data
from dedo.utils.args import get_args as dedo_get_args
from dedo.envs.deform_env import DeformEnv

#virtual wrapper class for all dedo environments
class DeformBulletEnv(DeformEnv):
    def __init__(self, render=False):
        dedo_args = dedo_get_args()
        dedo_args.task = self.get_task_name()
        dedo_args.viz = render
        dedo_args.debug = False
        
        super().__init__(args=dedo_args)
        #extend observation space with feature vertices
        # if hasattr(self.args, 'deform_true_loop_vertices'):
        #     n_feat = len(self.args.deform_true_loop_vertices[0])
        #     self.feat_lims = DeformEnv.WORKSPACE_BOX_SIZE*np.ones(n_feat*3)
        #     self.lims = np.concatenate([self.anchor_lims, self.feat_lims])
        #     self.observation_space = gym.spaces.Box(
        #         -1.0*self.lims, self.lims)

    def get_task_name(self):
        raise NotImplementedError
    
    def get_obs(self):
        obs, done = super().get_obs()
        # if hasattr(self.args, 'deform_true_loop_vertices'):
        #     #extend observation with feature vertice position
        #     _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        #     feat_pos = np.concatenate([vertex_positions[v] for v in self.args.deform_true_loop_vertices[0]])
        #     obs = np.concatenate([obs, feat_pos])
        return obs, done

class HangBagBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'HangBag'

class HangClothBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'HangCloth'

class ButtonSimpleBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'ButtonSimple'

class HoopBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'Hoop'

class LassoBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'Lasso'