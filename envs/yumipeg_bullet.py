import os

import numpy as np



import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import XmlBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene

# GOAL 1
GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
  0.57184653])
# obs in operational space
# note bullet quaternion is [x, y, z, w]
# GOAL_CART = np.array([ 0.46473501,  0.10293446,  0.10217953, -0.00858317,  0.69395054,  0.71995417,
#               0.00499788,  0.,          0.,          0.,          0.,          0.,      0.])
GOAL_CART = np.array([ 0.46473501,  0.10293446,  0.10217953, 0.69395054,  0.71995417, 0.00499788,  -0.00858317,  0.,          0.,          0.,          0.,          0.,      0.])

INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.]) #todo


ACTION_SCALE = 1e-3
# ACTION_SCALE = 1e-5
STATE_SCALE = 1

T = 200


class YumiRobot(XmlBasedRobot):
    def __init__(self):
        XmlBasedRobot.__init__(self, robot_name='yumi', action_dim=7, obs_dim=14, self_collision=True)
        self.model_xml = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'yumi_stable_vic_mjcf_bullet.xml')
        self._jnt_names = ['left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_joint_7']
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
        init_qpos = INIT 
        init_qvel = np.zeros(7)

        for i, jnt_name in enumerate(self._jnt_names):
            self.jdict[jnt_name].reset_current_position(
                init_qpos[i], init_qvel[i])
        

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for i, jnt_name in enumerate(self._jnt_names):
            self.jdict[jnt_name].set_motor_torque(a[i])

    def calc_state(self):
        #7x2 array
        jnt_state = np.array([self.jdict[jnt_name].current_position() for jnt_name in self._jnt_names])
        #a vector with first 7 dimensions as pos and remained as vel
        ob = jnt_state.transpose().flatten()
        return ob 
    
    def calc_potential(self):
        #dummy function for pybullet_envs
        return 0


class YumiPegBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, render=False):
        self.t = 0

        self.robot = YumiRobot()
        # self.reset_model()
        super().__init__(self.robot, render)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        ob = self._get_obs()

        pos = self.robot.parts["ball"].pose().xyz()
        ori = self.robot.parts["ball"].pose().orientation()

        # dist = pos - GOAL
        dist_t = pos - GOAL_CART[:3]
        # dist_r = rotations.quat_mul(GOAL_CART[3:7], rotations.quat_conjugate(ori))
        goal_mat = pybullet.getMatrixFromQuaternion(GOAL_CART[3:7])
        ori_mat = pybullet.getMatrixFromQuaternion(ori)
        #lets use rotation matrix to calculate the orientation difference because pybullet does not expose getQuaternionFromMatrix
        dist_r = np.trace(np.array(goal_mat).reshape((3, 3)).dot(np.array(ori_mat).reshape((3, 3)).T))

        reward_dist = -STATE_SCALE*(np.linalg.norm(dist_t) + 0.1*np.linalg.norm(dist_r))
        reward_ctrl = -ACTION_SCALE * np.square(a).sum()
        reward = reward_dist + reward_ctrl

        done = False
        self.t+=1
        if self.t >= T:
            done = True

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

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
    env = YumiPegBulletEnv(render=False)

    while True:
        env.reset()
        for i in range(200):
            env.step(env.action_space.sample())
            time.sleep(dt)
