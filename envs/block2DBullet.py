import os

import numpy as np

import gym.spaces

import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import XmlBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

############final values for the block exp#############
# GOAL = np.array([0, 0.5])
# INIT = np.array([-0.3, 0.8]) # pos1
# INIT = np.array([0.0, -0.1]) # pos2
# INIT = np.array([0.5, 0.3]) # pos3
# assumes <body name="blockx" pos="0.4 -0.6 0"> in xml
############final values for the block exp#############

#assumes <body name="blockx" pos="0 0 0"> in xml
OFFSET = np.array([0.4, -0.6])
# OFFSET_1 = np.array([0, -0.1])  # NFPPPO random init
OFFSET_1 = np.array([0, 0])         # NFPPPO fixed init
GOAL = np.array([0, 0.5])+OFFSET
# INIT = np.array([-0.3, 0.8])+OFFSET # pos1
INIT = np.array([-0.3, 0.8])+OFFSET+OFFSET_1
# INIT = np.array([0.0, -0.1])+OFFSET # pos2
# INIT = np.array([0.5, 0.3])+OFFSET # pos3

# ACTION_SCALE = 1e-3
# STATE_SCALE = 1
# TERMINAL_SCALE = 100
# EXP_SCALE = 2.
T = 200
POS_SCALE = 1
VEL_SCALE = 0.1
ACTION_SCALE = 1e-3
v = 2
w = 1
TERMINAL_STATE_SCALE = 10

SIGMA = np.array([0.05, 0.1]) # NFPPO

def cart_rwd_shape_1(d, v=1, w=1):

    alpha = 1e-5
    d_sq = d.dot(d)
    r = w*d_sq + v*np.log(d_sq + alpha) - v*np.log(alpha)
    assert (r >= 0)
    return r

def cart_rwd_func_1(x, f, terminal=False):
    '''
    This is for a regulation type problem, so x needs to go to zero.
    Magnitude of f has to be small
    :param x:
    :param f:
    :param g:
    :return:
    '''
    assert(x.shape==(4,))
    assert(f.shape==(2,))

    x_pos = x[:2]
    x_vel = x[2:]

    dx_pos = cart_rwd_shape_1(x_pos, v=v, w=w)
    dx_vel = x_vel.dot(x_vel)
    du = f.dot(f)

    reward_pos = -POS_SCALE*dx_pos
    if terminal:
        reward_pos = TERMINAL_STATE_SCALE * reward_pos

    reward_vel = -VEL_SCALE*dx_vel

    reward_state = reward_pos + reward_vel
    reward_action = -ACTION_SCALE*du
    reward = reward_state + reward_action
    rewards = np.array([reward_pos, reward_vel, reward_action])

    return reward, rewards

class Block2DRobot(XmlBasedRobot):
    def __init__(self):
        XmlBasedRobot.__init__(self, robot_name='blockx', action_dim=2, obs_dim=4, self_collision=True)
        self.model_xml = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'block2DForBullet.xml')
        self.doneLoading = 0
    
    def reset(self, bullet_client):
        self._p = bullet_client
        #print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(os.path.join(self.model_xml),
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                                pybullet.URDF_GOOGLEY_UNDEFINED_COLORS )
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(
                    os.path.join(self.model_xml, flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def robot_specific_reset(self, bullet_client):
        var = SIGMA ** 2
        cov = np.diag(var * np.ones(2))
        mu = INIT
        init_qpos = np.random.multivariate_normal(mu, cov)
        # init_qpos = INIT
        init_qvel = np.zeros(2)
        self.jdict["slidex"].reset_current_position(
            init_qpos[0], init_qvel[0])
        self.jdict["slidey"].reset_current_position(
            init_qpos[1], init_qvel[1])
        

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.jdict["slidex"].set_motor_torque(a[0])
        self.jdict["slidey"].set_motor_torque(a[1])

    def calc_state(self):
        x_pos, x_dot = self.jdict["slidex"].current_position()
        y_pos, y_dot = self.jdict["slidey"].current_position()
        ob=np.array([x_pos, y_pos, x_dot, y_dot])
        ob[:2]-=GOAL
        return ob 
    
    def calc_potential(self):
        #dummy function for pybullet_envs
        return 0


class Block2DBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, render=False):
        self.t = 0

        self.robot = Block2DRobot()
        # self.reset_model()
        super().__init__(self, self.robot, render)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        obs = self._get_obs()
        
        reward, rewards = cart_rwd_func_1(obs, a)
        done = False
        self.t+=1
        if self.t >= T:
            reward, rewards = cart_rwd_func_1(obs, a, terminal=True) 
            done = True
       
        return obs, reward, done, dict(reward_dist=np.sum(rewards[:2]), reward_ctrl=rewards[2])

    def reset(self):
        self.t = 0
        if (self.stateId >= 0):
            self._p.restoreState(self.stateId)
        r = MJCFBaseBulletEnv.reset(self)
        if (self.stateId < 0):
            self.stateId = self._p.saveState()
        return r
        

    def _get_obs(self):
        return self.robot.calc_state()
    

if __name__ == '__main__':
    "a test"
    import time
    dt = 1./240.
    env = Block2DBulletEnv(render=True)

    while True:
        env.reset()
        for i in range(200):
            env.step(env.action_space.sample())
            time.sleep(dt)
