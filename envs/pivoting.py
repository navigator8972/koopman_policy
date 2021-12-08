"""
A gym environment for a pivoting system reported in 
Rika Antonova and Silvia Cruciania, https://arxiv.org/abs/1703.00472
"""

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding


class PivotingEnv(gym.Env):
    """
    Description:
        A pole is held by a 1-DOF parallel gripper which can exert proper gripping forces 
        and tilt the finger to exploit inertia in pivoting its orientation.
        The interaction between finger and pole is modeled with Coulomb friction and linear deformable models.
        More details in https://arxiv.org/abs/1703.00472

    Observation:
        Type: Box(5)
        Num             Observation                                     Min             Max
        0               relative angle to target                        -180 deg        180 deg                  
        1               pole angular velocity (relative to gripper)                
        2               gripper orientation                             -90 deg         90 deg
        3               gripper angular velocity
        4               distance between two fingers                    0               d0
    
    Action:
        Type: Box(2)
        Num             Action                                          Min             Max
        0               acceleration of gripper tilting
        1               commanded finger tips distance                  0               d0
    
    Reward:
        -abs(obs[0])/range:         the range is a normalizing constant that the pole can attain, e.g. 2*pi if motion is not restricted
    
    Initial states:
        inital and target angular positions are chosen from [-pi/2, pi/2]
    """

    def __init__(self):
        self.g = 0          #assuming planar motion in the horizontal surface
        self.mass = 0.03    #mass of pole kg
        self.I = 9e-5       #inertia tensor of pole kg m^2
        self.r = 0.095      #distance between pivoting point and COG of pole
        self.d0 = 0.021     #rest distance bewteen two finger tips
        self.d0_lb = 0.017
        self.l = 0.35       #distance between gripper actuated joint and pivoting point, a setup suitable for baxter

        self.mu_v = 0.066*0.05 #0.066   #viscosity friction coefficient
        self.kmu_c_mean = 9.906*0.05  #Coulomb friction coefficient
        self.kmu_c = self.kmu_c_mean
        self.epsilon = 5e-2 #threshold of switching between sticking and sliping. adhoc value, not specified in the paper
        self.kgamma = 16    #static friction coeff together with deform stiffness. adhoc value, not specified in the paper

        self.dt = 1e-2     #time step for integration
        self.T = 1000       #number of steps
        self.ctrl_freq = 1 #internal integration steps
        self.gripper_latency_timeout = int(0.5 / (self.dt*self.ctrl_freq))
        self.gripper_latency_tick = 0
        self.gripper_latency_timeout_rand = 0
        self.pending_gripper_cmd = None

        self.obs_sample_timeout = 3
        self.obs_sample_tick = 0
        self.obs_curr = None

        #self.obs_gripper_offset = 0 #1.75

        self.phi_range = np.pi*2
        self.gripper_range = np.pi * 7 / 9.
        self.acc_limit = 18

        obs_high = np.array(
            [
                np.pi,
                np.finfo(np.float32).max,
                self.gripper_range/2, #+ self.obs_gripper_offset,
                np.finfo(np.float32).max,
                self.d0
            ]
        )

        obs_low = np.array(
            [
                -np.pi,
                np.finfo(np.float32).min,
                -self.gripper_range/2, #+ self.obs_gripper_offset,
                np.finfo(np.float32).min,
                self.d0_lb
            ]
        )

        act_high = np.array(
            [
                self.acc_limit,       
                self.d0                         #not realistic as there is always delay to command the desired finger distance
            ]
        )

        act_low = np.array(
            [
                -self.acc_limit,
                self.d0_lb
            ]
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """
        (I+mr^2+mlr*cos(tilt))*grp_acc + (I+mr^2)tilt_acc + mlr*sin(tilt)*grp_vel + mgr*cos(grp_pos+tilt_pos) = tau

        tau = -mu_v*tilt_vel - kmu_c*(d0-d)*sgn(tilt_vel)
        f_n = k(d0-d)
        """
        #constrain permitted acceleration
        action[0] = np.clip(action[0], -self.acc_limit, self.acc_limit)
        # action[1] = self.d0 # force a grip to test motion
        
        # common values
        I_plus_mrsquare = self.I + self.mass*self.r ** 2
        mlr = self.mass*self.l*self.r
        mgr = self.mass*self.g*self.r

        for _ in range(self.ctrl_freq):
            next_state = np.copy(self.state)
            #first lets integrate gripper kinematics, using semi-implicit euler
            #lets apply a velocity constraint for the consistency with real robot
            if np.abs(next_state[3]) > 2.1:
                acc_action = 0
            else:
                acc_action = action[0]
            next_state[3] += acc_action*self.dt
            next_state[2] += next_state[3]*self.dt
            
            if next_state[2] < -self.gripper_range/2 or next_state[2] > self.gripper_range/2:
                absolute_pivot_ang_vel = next_state[3] + next_state[1]
                next_state[3] = 0 
                next_state[1] = absolute_pivot_ang_vel
                
            next_state[2] = np.clip(next_state[2], -self.gripper_range/2, self.gripper_range/2)
            
            cmd_dist = self.state[4]

            #now integrate pole dynamics
            if np.abs(self.state[1]) < self.epsilon or self.state[4] < 0.0185:
                #sticking when the relative velocity is close to zero
                tau = mgr*np.cos(self.state[0]+self.state[2]) + mlr*np.sin(self.state[0])*self.state[3]**2 + (I_plus_mrsquare + mlr*np.cos(self.state[0]))*action[0]
                if self.state[4] >= 0.0185:
                    tau = np.clip(tau, -self.kgamma*(self.d0-cmd_dist), self.kgamma*(self.d0-cmd_dist))
                else:
                    next_state[1] = 0
            else:
                #sliping when the relative velocity is non-zero 
                tau = -self.mu_v*self.state[1]-self.kmu_c*(self.d0-cmd_dist)*np.sign(self.state[1])

            acc = tau - mgr*np.cos(self.state[0]+self.state[2]) - mlr*np.sin(self.state[0])*self.state[3]**2 - (I_plus_mrsquare + mlr*np.cos(self.state[0]))*action[0]
            acc = acc / I_plus_mrsquare

            next_state[1] += acc*self.dt
            next_state[0] += next_state[1]*self.dt
            #clip velocity
            if next_state[0] < -np.pi or next_state[0] > np.pi:
                next_state[1] = 0 
            next_state[0] = np.clip(next_state[0], -np.pi, np.pi)

            self.state = next_state
            #enforce action on finger distance, this may be subject to a dynamical process as well

            self.apply_grip_action(action[1])
            
        self.t += 1
        if self.pending_gripper_cmd is not None:
            self.gripper_latency_tick += 1
        self.obs_sample_tick+=1

        if self.t >= self.T:
            done = True
        else:
            done = False
        
        reward = -np.abs(self.state[0]-self.target)/self.phi_range

        #constraint penalty
        #avoid gripper going towards position limit
        #const_penalty = np.exp(2*(max(np.abs(next_state[2]) - self.gripper_range/2*0.8, 0) / (self.gripper_range/2*0.2) - 1))
        #avoid gripper moves close to the upperbound of velocity
        # const_penalty = 0.1*self.state[3]**2
        # reward -= const_penalty
        
        if np.abs(self.state[0]-self.target) < np.radians(3): # and np.abs(self.state[1]) < 0.1:
            #reach the goal region +-3 deg and almost static, bonus reward
            reward += 1
        return self.get_obs(), reward, done, {}
    
    def reset(self):
        self.state = np.array([#self.np_random.uniform()*np.radians(72),    #for the problematic half
            (self.np_random.uniform()-0.5)*2*np.radians(72), #angular pos of pole
            0,  # angular velocity of pole 
            0,  #(self.np_random.uniform()-0.5)*2*np.radians(85),  # gripper angular position
            0,  # gripper angular velocity 
            self.d0]    #finger tips distance
            )
        self.target = 0 #(self.np_random.uniform()-0.5)*2*np.radians(72)
        self.t = 0
        self.gripper_latency_tick = 0
        self.gripper_latency_timeout_rand = 0
        #we probably can randomize environment parameters here
        self.kmu_c = self.kmu_c_mean * (np.random.rand()*0.2+0.9)
        self.gripper_latency_timeout_rand = int(self.gripper_latency_timeout * (np.random.rand()-0.3))
        self.pending_gripper_cmd = None
        #initialize state for pole related observations
        self.obs_curr = np.copy(self.state[:2])
        return self.get_obs()
    
    def get_obs(self):
        obs = np.copy(self.state)
        obs[0]-=self.target    
        #for pole, the policy only access the real state at the rate of 30Hz
        if self.obs_sample_tick < self.obs_sample_timeout:
            obs[:2] = self.obs_curr
        else:
            self.obs_curr = np.copy(obs[:2])
            self.obs_sample_tick = 0
        #add noise to observations
        obs[0] += (np.random.rand()*2-1)*np.radians(1)
        obs[1] += (np.random.rand()*2-1)*np.radians(30)
        # obs[1] = 0
        #obs[2] += self.obs_gripper_offset
        # try to not close the loop on gripper observation
        obs[1] = 0
        obs[4] = 0.02

        return np.array(obs, dtype='f')

    def apply_grip_action(self, d):
        #here simulate the latency of applying gripping actions
        #this should be a nonblock call for ros action server
        if self.pending_gripper_cmd is None:
            self.pending_gripper_cmd = d
            self.gripper_latency_tick = 0

        if self.gripper_latency_tick > self.gripper_latency_timeout + self.gripper_latency_timeout_rand:
            #reset
            self.gripper_latency_tick = 0
            #we may also apply some randomness to the threshold for the next timeout
            #apply the desired distance, comment out to disable it
            self.state[4] = self.pending_gripper_cmd
            self.pending_gripper_cmd = None
        else:
            #the command will be ignored if the action is blocked
            pass
        #we need to update this in the main step loop if apply_grip_action is not invoked at every step
        # self.gripper_latency_tick += 1
        return

    def render(self, mode="human"):
        """
        2D plot to illustrate planar 2-link underactuated system
        setups referred to cartpole
        """
        screen_width=800
        screen_height=800

        phys_pixel_ratio = 700   #0.8 m corresponds to 600 pixels

        gripper_link_length = self.l * phys_pixel_ratio
        gripper_link_width = self.l/2 * phys_pixel_ratio

        pole_link_length = self.r*2 * phys_pixel_ratio
        pole_link_width = self.r/2 * phys_pixel_ratio


        if self.viewer is None:
            from gym.envs.classic_control import rendering

            base_trans = rendering.Transform(translation=(200, 400))
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #gripper polygon
            l, r, t, b = -gripper_link_length/2, gripper_link_length/2, gripper_link_width/2, -gripper_link_width/2
            gripper_link = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

            self.gripper_link_trans = rendering.Transform()
            gripper_link.add_attr(rendering.Transform(translation=(gripper_link_length/2, 0)))
            gripper_link.add_attr(self.gripper_link_trans)
            gripper_link.add_attr(base_trans)

            self.viewer.add_geom(gripper_link)

            #pole
            l, r, t, b = -pole_link_length/2, pole_link_length/2, pole_link_width/2, -pole_link_width/2
            pole_link = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole_link.set_color(0.8, 0.6, 0.4)
            
            self.pole_trans = rendering.Transform()
            pole_link.add_attr(rendering.Transform(translation=(pole_link_length/2, 0)))
            pole_link.add_attr(self.pole_trans)
            pole_link.add_attr(rendering.Transform(translation=(gripper_link_length, 0)))
            pole_link.add_attr(self.gripper_link_trans)
            pole_link.add_attr(base_trans)

            self.viewer.add_geom(pole_link)

            #pole target
            pole_link_target = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole_link_target._color.vec4 = (0.8, 0.6, 0.4, 0.3) #try to use transparency
            self.pole_link_target_trans = rendering.Transform()
            pole_link_target.add_attr(rendering.Transform(translation=(pole_link_length/2, 0)))
            pole_link_target.add_attr(self.pole_link_target_trans)
            pole_link_target.add_attr(rendering.Transform(translation=(gripper_link_length, 0)))
            pole_link_target.add_attr(self.gripper_link_trans)
            pole_link_target.add_attr(base_trans)

            self.viewer.add_geom(pole_link_target)

            #axle
            self.axle = rendering.make_circle(gripper_link_width/8)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.axle.add_attr(base_trans)
            self.viewer.add_geom(self.axle)

            #gripper pressure represented by a circle at axle
            r = gripper_link_width/4
            press_circle = rendering.make_circle(r)
            press_circle._color.vec4=(0.8, 0.2, 0.2, 0.8)
            self.press_circle_scale = rendering.Transform()
            press_circle.add_attr(self.press_circle_scale)
            press_circle.add_attr(rendering.Transform(translation=(gripper_link_length, 0)))
            press_circle.add_attr(self.gripper_link_trans)
            press_circle.add_attr(base_trans)
            self.viewer.add_geom(press_circle)
        

        if self.state is None:
            return None
        
        self.gripper_link_trans.set_rotation(self.state[2])
        self.pole_trans.set_rotation(self.state[0])
        self.pole_link_target_trans.set_rotation(self.target)
        #set scale of press circle
        s = 1.1-self.state[4] / float(self.d0) 
        self.press_circle_scale.set_scale(s, s)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
if __name__ == '__main__':
    "a test"
    import time
    dt = 1./50.
    env = PivotingEnv()
    env.reset()
   
    for i in range(1000):
        env.step(env.action_space.sample())
        # time.sleep(dt)
        env.render()
    
    env.close()
