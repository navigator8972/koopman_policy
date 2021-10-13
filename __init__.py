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
     id='PivotingEnv-v0',
     entry_point='koopman_policy.envs:PivotingEnv',
     max_episode_steps=400,
)

register(
     id='HangGarmentBulletEnv-v1',
     entry_point='koopman_policy.envs:HangGarmentBulletEnv',
     max_episode_steps=400,
)

register(
     id='ButtonSimpleBulletEnv-v1',
     entry_point='koopman_policy.envs:ButtonSimpleBulletEnv',
     max_episode_steps=200,
)


# register(
#      id='YumiPeg-v0',
#      entry_point='koopman_policy.envs:YumiPegEnv',
#      max_episode_steps=1000,
# )