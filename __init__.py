from gym.envs.registration import register

register(
     id='Block2D-v0',
     entry_point='koopman_policy.envs:Block2DEnv',
     max_episode_steps=1000,
)