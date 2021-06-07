import torch
import torch.nn as nn
from torch.nn import functional as F
import gym

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.sampler import RaySampler, LocalSampler


import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy 

@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def ppo_peginhole(ctxt=None, seed=1, policy_type='koopman'):
    """Train PPO with PegInHole environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv('Block2D-v0')
    # env = GymEnv('YumiPeg-v0')

    trainer = Trainer(ctxt)

    hidden_size = 32

    if policy_type == 'vanilla':
        print('Using Vanilla NN Policy')
        policy = GaussianMLPPolicy(env.spec,
                                hidden_sizes=[hidden_size, hidden_size],
                                hidden_nonlinearity=F.relu,
                                output_nonlinearity=None)
    else:
        in_dim = env.spec.observation_space.flat_dim
        hidden_dim = hidden_size
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            print('Using Koopman Policy')
            residual = None
        else:
            print('Using Koopman NN Policy')
            residual = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )            

        
        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,
            T=5,
            phi='FCNN',
            residual=residual,
            init_std=1.0,
            use_state_goal='state'
        )

        #fix the goal at origin
        policy.set_state_goal_learnable(state_goal=None, learnable=False)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    # algo = PPO(env_spec=env.spec,
    #            policy=policy,
    #            value_function=value_function,
    #            discount=0.99,
    #            center_adv=False)
    
    sampler = LocalSampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               lr_clip_range=0.2,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1500, plot=True)

#[1, 21, 52, 251, 521]
#[2, 12, 51, 125, 512]
seed = 251
ppo_peginhole(seed=seed, policy_type='vanilla')
ppo_peginhole(seed=seed, policy_type='koopman')
ppo_peginhole(seed=seed, policy_type='koopman_residual')