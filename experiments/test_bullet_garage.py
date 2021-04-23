import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import gym

from garage import wrap_experiment
# from garage.envs import GymEnv, normalize
from garage.envs.bullet import BulletEnv
from garage.envs import normalize

from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy, KoopmanLQRRLParam
from koopman_policy.koopmanlqr_sac_garage import KoopmanLQRSAC
from koopman_policy.koopmanlqr_ppo_garage import KoopmanLQRPPO

from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC

from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.distributions import TanhNormal

@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def koopmanlqr_sac_bullet_tests(ctxt=None, seed=1, policy_type='koopman', policy_horizon=5):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    # env = gym.make('CartPoleBulletEnv-v1', renders=False)
    env = normalize(BulletEnv('InvertedPendulumBulletEnv-v0'))
    # env = BulletEnv('ReacherBulletEnv-v0')
    # env = normalize(BulletEnv('HalfCheetahBulletEnv-v0'))

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
            T=policy_horizon,
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

        koopman_param = KoopmanLQRRLParam(
            least_square_fit_coeff=-1,
            koopman_fit_coeff=1,
            koopman_fit_coeff_errbound=-1,
            koopman_fit_optim_lr=-1,
            koopman_fit_n_itrs=1,
            koopman_fit_mat_reg_coeff=0.1,
            koopman_recons_coeff=1
        )

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
            koopman_param=koopman_param
            )

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=50, batch_size=1000, plot=False)
    return


@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def koopmanlqr_ppo_bullet_tests(ctxt=None, seed=1, policy_type='koopman', policy_horizon=5):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    # env = gym.make('CartPoleBulletEnv-v1', renders=False)
    env = normalize(BulletEnv('InvertedPendulumBulletEnv-v0'))
    # env = BulletEnv('ReacherBulletEnv-v0')

    #original hidden size 256
    hidden_size = 32

    if policy_type == 'vanilla':
        print('Using Vanilla NN Policy')
        policy = GaussianMLPPolicy(env.spec,
                                hidden_sizes=[hidden_size, hidden_size],
                                hidden_nonlinearity=F.relu,
                                output_nonlinearity=None)

        #disable all koopman relevant parameters so the PPO will be vanilla as well
        koopman_param = KoopmanLQRRLParam(
            least_square_fit_coeff=-1,
            koopman_fit_coeff=-1,
            koopman_fit_coeff_errbound=-1,
            koopman_fit_optim_lr=-1,
            koopman_fit_n_itrs=-1,
            koopman_fit_mat_reg_coeff=-1,
            koopman_recons_coeff=-1
        )

    else:
        in_dim = env.spec.observation_space.flat_dim
        hidden_dim = hidden_size
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            print('Using Koopman Policy')
            residual = None
        else:
            print('Using Koopman NN Policy')
            residual = koopman_policy.koopman_lqr.FCNN(in_dim, out_dim, [hidden_dim, hidden_dim], hidden_nonlinearity=nn.ReLU)
          

        
        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,
            T=policy_horizon,
            phi=[hidden_dim, hidden_dim],
            residual=residual,
            init_std=1.0,
            use_state_goal=False
        )

        #fix the goal at origin
        #policy.set_state_goal_learnable(state_goal=None, learnable=False)

        koopman_param = KoopmanLQRRLParam(
            least_square_fit_coeff=-1,
            koopman_fit_coeff=1,
            koopman_fit_coeff_errbound=-1,
            koopman_fit_optim_lr=-1,
            koopman_fit_n_itrs=-1,
            koopman_fit_mat_reg_coeff=0.1,
            koopman_recons_coeff=1
        )

    #shared settings    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    
    sampler = LocalSampler(agents=policy,
                        envs=env,
                        max_episode_length=500)
                        #env.spec.max_episode_length)
    if policy_type=='vanilla':
        algo = PPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                discount=0.99,
                gae_lambda=0.95,
                lr_clip_range=0.2,
                center_adv=False)
    else:
        algo = KoopmanLQRPPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               lr_clip_range=0.2,
               center_adv=False,
               #use extra koopman_param, could be dummy for vanilla PPO and policy
               koopman_param=koopman_param
               )

    trainer.setup(algo, env)
    trainer.train(n_epochs=50, batch_size=5000, plot=False)
    return

seeds = [1, 21, 52, 251, 521]
#seeds = [251, 521]
#[2, 12, 51, 125, 512]
for seed in seeds: 
#     koopmanlqr_sac_bullet_tests(seed=seed, policy_type='vanilla')
#     koopmanlqr_sac_bullet_tests(seed=seed, policy_type='koopman', policy_horizon=5)
#     koopmanlqr_sac_bullet_tests(seed=seed, policy_type='koopman_residual', policy_horizon=5)
      koopmanlqr_ppo_bullet_tests(seed=seed, policy_type='vanilla')
      koopmanlqr_ppo_bullet_tests(seed=seed, policy_type='koopman')
      koopmanlqr_ppo_bullet_tests(seed=seed, policy_type='koopman_residual', policy_horizon=5)
#test the impact of time horizon
#for seed in seeds:
#    for h in [2, 5, 8, 12, 15]:
        #koopmanlqr_sac_bullet_tests(seed=seed, policy_type='koopman', policy_horizon=h)
        #koopmanlqr_ppo_bullet_tests(seed=seed, policy_type='koopman', policy_horizon=h)
#koopmanlqr_ppo_bullet_tests(seed=1, policy_type='koopman', policy_horizon=5)
#koopmanlqr_sac_bullet_tests(seed=1, policy_type='koopman', policy_horizon=5)
