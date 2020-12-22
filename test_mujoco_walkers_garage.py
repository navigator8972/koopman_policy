import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gym

from garage import wrap_experiment
from garage.envs import GymEnv, normalize

from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy 

@wrap_experiment
def ppo_mujoco_walkers(ctxt=None, seed=1):
    """Train PPO with Mujoco walker environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = normalize(GymEnv('HalfCheetah-v2'))

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=F.relu,
                               output_nonlinearity=None)

    # in_dim = env.spec.observation_space.flat_dim
    # hidden_dim = 32
    # out_dim = env.spec.action_space.flat_dim

    # residual = nn.Sequential(
    #     nn.Linear(in_dim, hidden_dim),
    #     nn.GELU(),
    #     nn.Linear(hidden_dim, hidden_dim),
    #     nn.GELU(),
    #     nn.Linear(hidden_dim, out_dim),
    # )
    # policy = GaussianKoopmanLQRPolicy(
    #     env_spec=env.spec,
    #     k=4,
    #     T=5,
    #     phi='FCNN',
    #     residual=residual,
    #     # normal_distribution_cls=Normal,
    #     init_std=1.0,
    # )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=F.relu,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               lr_clip_range=0.2,
               center_adv=True)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1500, plot=True)
    return

from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC

from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.distributions import TanhNormal

@wrap_experiment(snapshot_mode='none')
def sac_mujoco_walkers(ctxt=None, seed=1, policy_type='koopman'):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    # env = normalize(GymEnv('HalfCheetah-v2'))
    # env = normalize(GymEnv('Walker2d-v2'))
    env = normalize(GymEnv('Ant-v2'))

    if policy_type == 'vanilla':
        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[32, 32],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )
    else:
        in_dim = env.spec.observation_space.flat_dim
        hidden_dim = 32
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            residual = None
        else:
            residual = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )            

        
        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,
            T=8,
            phi='FCNN',
            residual=residual,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
        )

    #original hidden size 256
    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[32, 32],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=FragmentWorker)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              sampler=sampler,
              gradient_steps_per_itr=1000,
              max_episode_length_eval=1000,
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
    trainer.train(n_epochs=200, batch_size=1000, plot=False)
    return

# ppo_mujoco_walkers(seed=521)
sac_mujoco_walkers(seed=521, policy_type='vanilla')
sac_mujoco_walkers(seed=521, policy_type='koopman')
sac_mujoco_walkers(seed=521, policy_type='koopman_residual')