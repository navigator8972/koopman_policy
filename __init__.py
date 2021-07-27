from gym.envs.registration import register

# register(
#      id='Block2D-v0',
#      entry_point='koopman_policy.envs:Block2DEnv',
#      max_episode_steps=200,
# )

register(
     id='Block2DBulletEnv-v0',
     entry_point='koopman_policy.envs:Block2DBulletEnv',
     max_episode_steps=200,
)

register(
     id='HangBagBulletEnv-v0',
     entry_point='koopman_policy.envs:HangBagBulletEnv',
     max_episode_steps=400,
)

register(
     id='HangClothBulletEnv-v0',
     entry_point='koopman_policy.envs:HangClothBulletEnv',
     max_episode_steps=400,
)


# register(
#      id='YumiPeg-v0',
#      entry_point='koopman_policy.envs:YumiPegEnv',
#      max_episode_steps=1000,
# )