import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

############final values for the block exp#############
# GOAL = np.array([0, 0.5])
# INIT = np.array([-0.3, 0.8]) # pos1
# INIT = np.array([0.0, -0.1]) # pos2
# INIT = np.array([0.5, 0.3]) # pos3
# assumes <body name="blockx" pos="0.4 -0.6 0"> in xml
############final values for the block exp#############

#assumes <body name="blockx" pos="0 0 0"> in xml
OFFSET = np.array([0.4, -0.6])
# OFFSET = np.array([0, 0])
GOAL = np.array([0, 0.5])+OFFSET
INIT = np.array([-0.3, 0.8])+OFFSET # pos1
# INIT = np.array([0.0, -0.1])+OFFSET # pos2
# INIT = np.array([0.5, 0.3])+OFFSET # pos3

ACTION_SCALE = 1e-3
# ACTION_SCALE = 1e-5
STATE_SCALE = 1
TERMINAL_SCALE = 1000
T = 100
EXP_SCALE = 2.

class Block2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.t = 0

        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'block2D.xml')
        mujoco_env.MujocoEnv.__init__(self, fullpath, 1)
        
        self.reset_model()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        dist = ob[:2]
        reward_dist = -STATE_SCALE*np.linalg.norm(dist)
        reward_ctrl = -ACTION_SCALE * np.square(a).sum()
        reward = reward_dist + reward_ctrl

        done = False
        self.t+=1
        if self.t >= T:
            done = True
       
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = 4.0

    def reset_model(self):
        init_qpos = INIT + (np.random.rand(2) * 0.2 - np.array([0.1, 0.1]))*1
        init_qvel = np.zeros(2)
        self.set_state(init_qpos, init_qvel)
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        ob=np.concatenate([
            # self.model.data.qpos.flat[:7],
            # self.model.data.qvel.flat[:7],
            self.sim.data.qpos.flat[:2],
            self.sim.data.qvel.flat[:2],
            # self.get_body_com("blocky")[:2],
        ])
        ob[:2]-=GOAL
        return ob 
    
    def close(self):
        return True

# from pybullet.envs.env_bases import MJCFBaseBulletEnv
# from pybullet.envs.robot_bases import  MJCFBasedRobot

# class Block2DBulletEnv(MJCFBaseBulletEnv, utils.EzPickle):
#     def __init__(self):
#         utils.EzPickle.__init__(self)

#         self.t = 0

#         fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'block2D.xml')
#         mujoco_env.MujocoEnv.__init__(self, fullpath, 1)
        
#         self.reset_model()

#     def step(self, a):
#         self.do_simulation(a, self.frame_skip)
#         ob = self._get_obs()
#         dist = ob[:2]
#         reward_dist = -STATE_SCALE*np.linalg.norm(dist)
#         reward_ctrl = -ACTION_SCALE * np.square(a).sum()
#         reward = reward_dist + reward_ctrl

#         done = False
#         self.t+=1
#         if self.t >= T:
#             done = True
       
#         return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 0
#         # self.viewer.cam.trackbodyid = -1
#         # self.viewer.cam.distance = 4.0

#     def reset_model(self):
#         init_qpos = INIT + (np.random.rand(2) * 0.2 - np.array([0.1, 0.1]))*1
#         init_qvel = np.zeros(2)
#         self.set_state(init_qpos, init_qvel)
#         self.t = 0
#         return self._get_obs()

#     def _get_obs(self):
#         ob=np.concatenate([
#             # self.model.data.qpos.flat[:7],
#             # self.model.data.qvel.flat[:7],
#             self.sim.data.qpos.flat[:2],
#             self.sim.data.qvel.flat[:2],
#             # self.get_body_com("blocky")[:2],
#         ])
#         ob[:2]-=GOAL
#         return ob 
    
#     def close(self):
#         return True