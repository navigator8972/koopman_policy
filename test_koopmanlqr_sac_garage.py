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

import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy 
from koopman_policy.koopmanlqr_sac_garage import KoopmanLQRSAC

from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC

from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.distributions import TanhNormal

@wrap_experiment(snapshot_mode='none')
def koopmanlqr_sac_mujoco_tests(ctxt=None, seed=1, policy_type='koopman'):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    # env = normalize(GymEnv('HalfCheetah-v2'))
    # env = normalize(GymEnv('Hopper-v2'))
    # env = normalize(GymEnv('InvertedPendulum-v2'))
    # env = normalize(GymEnv('Reacher-v2'))
    env = GymEnv('Block2D-v0')
        
    #original hidden size 256
    hidden_size = 32

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[hidden_size, hidden_size],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[hidden_size, hidden_size],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))


    if policy_type == 'vanilla':
        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[hidden_size, hidden_size],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )

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
            max_episode_length_eval=500,
            replay_buffer=replay_buffer,
            min_buffer_size=1e4,
            target_update_tau=5e-3,
            discount=0.99,
            buffer_batch_size=256,
            reward_scale=1.,
            steps_per_epoch=1)
    else:
        in_dim = env.spec.observation_space.flat_dim
        hidden_dim = hidden_size
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            residual = None
        else:
            residual = koopman_policy.koopman_lqr.FCNN(in_dim, out_dim, [hidden_dim, hidden_dim], hidden_nonlinearity=nn.ReLU)
        
        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,   #use the same size of koopmanv variable
            T=5,
            phi=[hidden_dim, hidden_dim],
            residual=residual,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            use_state_goal=False,   #non regularization task, should be flexible?
        )

        sampler = LocalSampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length,
                        worker_class=FragmentWorker)

        sac = KoopmanLQRSAC(env_spec=env.spec,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            sampler=sampler,
            gradient_steps_per_itr=1000,
            max_episode_length_eval=500,
            replay_buffer=replay_buffer,
            min_buffer_size=1e4,
            target_update_tau=5e-3,
            discount=0.99,
            buffer_batch_size=256,
            reward_scale=1.,
            steps_per_epoch=1,
            #new params
            least_square_fit_coeff=-1,
            koopman_fit_coeff=10,
            koopman_fit_coeff_errbound=-1,
            koopman_recons_coeff=-1
            )

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=300, batch_size=1000, plot=False)
    return

#[1, 21, 52, 251, 521]
#[2, 12, 51, 125, 512]
seed = 1
# koopmanlqr_sac_mujoco_tests(seed=seed, policy_type='vanilla')
koopmanlqr_sac_mujoco_tests(seed=seed, policy_type='koopman')
koopmanlqr_sac_mujoco_tests(seed=seed, policy_type='koopman_residual')
