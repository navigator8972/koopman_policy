import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
# import gym

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
#import garage.envs as garage_envs
# from garage.envs import GymEnv

from garage.experiment.deterministic import set_seed
# from garage.sampler import RaySampler

# from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC

from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.distributions import TanhNormal

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
# from softgym.utils.normalized_env import normalize

import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy 


@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def sac_softgym(ctxt=None, seed=1, policy_type='koopman', args=None):
    set_seed(seed)

    assert(args is not None)
    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    # env_kwargs['num_variations'] = args.num_variations
    env_kwargs['num_variations'] = 1
    # env_kwargs['render'] = True
    env_kwargs['render'] = True
    # env_kwargs['headless'] = args.headless
    env_kwargs['headless'] = 1
    
    # use only point cloud or keypoint representation
    env_kwargs['observation_mode'] = 'key_point'

    softgym_env = SOFTGYM_ENVS[args.env_name](**env_kwargs)
    #need to assign max_episode_length to prevent complaints from samplers
    env = normalize(GymEnv(softgym_env, max_episode_length=softgym_env.horizon))
    env.reset()

    trainer = Trainer(snapshot_config=ctxt)

    #original hidden size 256
    hidden_size = 32

    if policy_type == 'vanilla':
        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[hidden_size, hidden_size],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )
    else:
        in_dim = env.spec.observation_space.flat_dim
        hidden_dim = hidden_size
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            residual = None
        else:
            # residual = nn.Sequential(
            #     nn.Linear(in_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, out_dim),
            # )            
            residual = koopman_policy.koopman_lqr.FCNN(in_dim, out_dim, [hidden_dim, hidden_dim], hidden_nonlinearity=nn.ReLU)
        
        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,
            T=5,
            phi=[hidden_dim, hidden_dim],
            residual=residual,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            use_state_goal=False    #non regularization task, should be more flexible?
        )

    
    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[hidden_size, hidden_size],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[hidden_size, hidden_size],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           n_workers=1,                 #be conservative here...
                           worker_class=DefaultWorker)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=1000,
              max_episode_length_eval=100,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=1)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=100, batch_size=1000, plot=False)
    return

#[1, 21, 52, 251, 521]
#[2, 12, 51, 125, 512]
def main():
    seed = 521

    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')

    args = parser.parse_args()

    sac_softgym(seed=seed, policy_type='vanilla', args=args)
    # sac_softgym(seed=seed, policy_type='koopman', args=args)
    # sac_softgym(seed=seed, policy_type='koopman_residual', args=args)
    return

if __name__ == '__main__':
    main()
