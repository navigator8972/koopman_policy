
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize

from garage.experiment.deterministic import set_seed

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy, KoopmanLQRRLParam
from koopman_policy.koopmanlqr_ppo_garage import KoopmanLQRPPO

from garage.sampler import DefaultWorker, LocalSampler, VecWorker, MultiprocessingSampler
from garage.torch import set_gpu_mode

@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def koopmanlqr_ppo_pivoting_tests(ctxt=None, seed=1, policy_type='koopman', policy_horizon=5):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    env = normalize(GymEnv('PivotingEnv-v0'))

    # print(env.spec.observation_space)
    # print(env.spec.observation_space.flat_dim)

    #need a separate seed for gym environment for full determinism
    env.seed(seed)
    env.action_space.seed(seed)
    #original hidden size 256
    hidden_size = 32

    if policy_type == 'vanilla':
        print('Using Vanilla NN Policy')
        policy = GaussianMLPPolicy(env.spec,
                                hidden_sizes=[hidden_size, hidden_size],
                                hidden_nonlinearity=torch.relu,
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
        out_dim = env.spec.action_space.flat_dim

        if policy_type == 'koopman':
            print('Using Koopman Policy')
            residual = None
        else:
            print('Using Koopman NN Policy')
            residual = koopman_policy.koopman_lqr.FCNN(in_dim, out_dim, [hidden_size, hidden_size], hidden_nonlinearity=nn.ReLU)

        policy = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=4,
            T=policy_horizon,
            phi=[hidden_size, hidden_size],
            residual=residual,
            init_std=1.0,
            use_state_goal='latent'
        )

        # policy.set_state_goal_learnable(state_goal=None, learnable=False) #fixed origin goal

        #fix the goal at origin
        #policy.set_state_goal_learnable(state_goal=None, learnable=False)
        koopman_param = KoopmanLQRRLParam(
                least_square_fit_coeff=-1,    #use least square
                koopman_fit_coeff=1,
                koopman_fit_coeff_errbound=-1,
                koopman_fit_optim_lr=-1,
                koopman_fit_n_itrs=-1,
                koopman_fit_mat_reg_coeff=-1,
                koopman_recons_coeff=1,
                koopman_nonnn_lr=0.01
            )

        # koopman_param._koopman_recons_coeff = -1

    #shared settings    
    #need a separate hiddenzie for MLP because of the experience of using linearly parameterized approximator
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(hidden_size, hidden_size),
                                              hidden_nonlinearity=F.relu,
                                              output_nonlinearity=None)

    sampler = MultiprocessingSampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length,
                        worker_class=DefaultWorker)

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
    # if torch.cuda.is_available():
    #     set_gpu_mode(True)
    # else:
    #     set_gpu_mode(False)
    # algo.to()
    trainer.setup(algo, env)
    trainer.train(n_epochs=20, batch_size=40000, plot=True)
    return


def main():
    #seeds = [1, 21, 52, 251, 521]
    seeds = [1]
    for seed in seeds:
        koopmanlqr_ppo_pivoting_tests(seed=seed, policy_type='vanilla')
        koopmanlqr_ppo_pivoting_tests(seed=seed, policy_type='koopman')
        koopmanlqr_ppo_pivoting_tests(seed=seed, policy_type='koopman_residual')


if __name__ == '__main__':
    main()
