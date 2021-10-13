#wrapper for DEDO environments
#these are necessary because garage do not call gym.make with kwargs
import numpy as np
import gym

from dedo.utils.mesh_utils import get_mesh_data
from dedo.utils.args import get_args as dedo_get_args
from dedo.envs.deform_env import DeformEnv
from dedo.utils.anchor_utils import command_anchor_velocity

#virtual wrapper class for all dedo environments
class DeformBulletEnv(DeformEnv):
    def __init__(self, render=False):
        dedo_args = dedo_get_args()
        dedo_args.robot = 'anchor'
        dedo_args.task = self.get_task_name()
        dedo_args.viz = render
        dedo_args.debug = False
        dedo_args.cam_resolution = -1
        dedo_args.version = 1
        #this is a flag to indicate if the environment is used for visualization
        #this will not be touched for regular training and will only be manually switched on in visualization
        self.viz_mode = 0
        
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
        return np.array(obs, dtype='f'), done
    
    def step(self, action):
        if self.viz_mode:
            next_obs, reward, done, info = self.step_without_after_done(action, unscaled=False)
        else:
            next_obs, reward, done, info = super().step(action, unscaled=False)
        if not 'is_success' in info:
            #complete info for consistent return
            info['is_success'] = False
            info['final_reward'] = 0
        
        return next_obs, reward, done, info
    
    def step_without_after_done(self, action, unscaled=False):
        #overrided step without doing done stepSimulation
        #this is used to separate afterdone steps for visualization
                # action is num_anchors x 3 for 3D velocity for anchors/grippers;
        # assume action in [-1,1], we convert to [-MAX_ACT_VEL, MAX_ACT_VEL].

        if self.args.debug:
            print('action', action)
        if not unscaled:
            assert self.action_space.contains(action)
            assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
            if self.robot is None:  # velocity control for anchors
                action *= DeformEnv.MAX_ACT_VEL
            else:
                action *= DeformEnv.WORKSPACE_BOX_SIZE  # pos control for robots
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        raw_force_accum = 0.0
        for sim_step in range(self.args.sim_steps_per_action):
            for i in range(self.num_anchors):
                if self.args.robot == 'anchor':
                    curr_force = command_anchor_velocity(
                        self.sim, self.anchor_ids[i], action[i])
                    raw_force_accum += np.linalg.norm(curr_force)
            if self.args.robot != 'anchor':  # robot Cartesian position control
                self.do_robot_action(action)
            self.sim.stepSimulation()
        # Get next obs, reward, done.
        next_obs, done = self.get_obs()
        reward = self.get_reward()
        if done:  # if terminating early use reward from current step for rest
            reward *= (self.max_episode_len - self.stepnum)
        done = (done or self.stepnum >= self.max_episode_len)
        info = {}

        self.episode_reward += reward  # update episode reward
        self.stepnum += 1
        
        return next_obs, reward, done, info

class HangBagBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'HangBag'

class HangGarmentBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'HangGarment'

class ButtonSimpleBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'Button'

class HoopBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'Hoop'

class LassoBulletEnv(DeformBulletEnv):
    def get_task_name(self):
        return 'Lasso'