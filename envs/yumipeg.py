import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from gym.envs.robotics import rotations

# GOAL 1
GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
  0.57184653])
# obs in operational space
GOAL_CART = np.array([ 0.46473501,  0.10293446,  0.10217953, -0.00858317,  0.69395054,  0.71995417,
              0.00499788,  0.,          0.,          0.,          0.,          0.,      0.])

INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.]) #todo


ACTION_SCALE = 1e-3
# ACTION_SCALE = 1e-5
STATE_SCALE = 1

T = 200


class YumiPegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

        self.t = 0

        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'yumi_stable_vic_mjcf.xml')
        mujoco_env.MujocoEnv.__init__(self, fullpath, 1)
        
        self.reset_model()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        pos = self.get_body_com("ball")
        ori = self.data.get_body_xquat("ball")

        # dist = pos - GOAL
        dist_t = pos - GOAL_CART[:3]
        dist_r = rotations.quat_mul(GOAL_CART[3:7], rotations.quat_conjugate(ori))
        reward_dist = -STATE_SCALE*(np.linalg.norm(dist_t) + 0.1*np.linalg.norm(dist_r))
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
        init_qpos = INIT 
        init_qvel = np.zeros(7)
        self.set_state(init_qpos, init_qvel)
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            # self.model.data.qpos.flat[:7],
            # self.model.data.qvel.flat[:7],
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            # self.get_body_com("blocky")[:2],
        ])
    
    def close(self):
        return True