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

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

import koopman_policy
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy, KoopmanLQRRLParam
from koopman_policy.koopmanlqr_policy_garage import GaussianKoopmanMLPValueFunction, ContinuousKoopmanMLPQFunction
from koopman_policy.koopmanlqr_sac_garage import KoopmanLQRSAC
from koopman_policy.koopmanlqr_ppo_garage import KoopmanLQRPPO

from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker, LocalSampler, VecWorker, RaySampler, MultiprocessingSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC

from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.distributions import TanhNormal

@wrap_experiment
def koopmanlqr_sac_bullet_tests(ctxt=None, config=None):
    assert(config is not None)
    seed = config['seed']
    policy_type = config['policy_type']
    policy_horizon = config['koopman_horizon']

    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    env = normalize(BulletEnv(config['env_name']))

    if policy_horizon is None:
        policy_horizon = config['koopman_horizon']
    if 'num_evaluation_episodes' in config:
        num_evaluation_episodes = config['num_evaluation_episodes']
    else:
        num_evaluation_episodes = 1

    #need a separate seed for gym environment for full determinism
    env.seed(seed)
    env.action_space.seed(seed)
    #original hidden size 256
    hidden_size = config['hidden_size']

    if config['valuefunc_type'] == 'koopman':
        if policy_type != 'koopman':
            k = 4
            T = 5
        else:
            k = config['koopman_size']
            T = policy_horizon
        
            #use default koopman policy param
        koopman1 = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=k,   #use the same size of koopmanv variable
            T=T,
            phi=[hidden_size, hidden_size],
            residual=None,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            use_state_goal='latent')
        koopman2 = GaussianKoopmanLQRPolicy(
            env_spec=env.spec,
            k=k,   #use the same size of koopmanv variable
            T=T,
            phi=[hidden_size, hidden_size],
            residual=None,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            use_state_goal='latent')
        
        qf1 = ContinuousKoopmanMLPQFunction(koopman1._kpm_ctrl, 
                                    env_spec=env.spec,
                                    hidden_sizes=[hidden_size, hidden_size],
                                    hidden_nonlinearity=torch.relu)
        qf2 = ContinuousKoopmanMLPQFunction(koopman2._kpm_ctrl, 
                            env_spec=env.spec,
                            hidden_sizes=[hidden_size, hidden_size],
                            hidden_nonlinearity=torch.relu)
    else:
        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[hidden_size, hidden_size],
                                    hidden_nonlinearity=torch.relu)

        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[hidden_size, hidden_size],
                                    hidden_nonlinearity=torch.relu)

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
            k=config['koopman_size'],   #use the same size of koopmanv variable
            T=policy_horizon,
            phi=[hidden_dim, hidden_dim],
            residual=residual,
            normal_distribution_cls=TanhNormal,
            init_std=1.0,
            use_state_goal='latent'
        )

        koopman_param = KoopmanLQRRLParam(
                least_square_fit_coeff=config['least_square_fit_coeff'],    #use least square
                koopman_fit_coeff=config['koopman_fit_coeff'],
                koopman_fit_coeff_errbound=config['koopman_fit_coeff_errbound'],
                koopman_fit_optim_lr=config['koopman_fit_optim_lr'],
                koopman_fit_n_itrs=config['koopman_fit_n_itrs'],
                koopman_fit_mat_reg_coeff=config['koopman_fit_mat_reg_coeff'],
                koopman_recons_coeff=config['koopman_recons_coeff'],
                koopman_nonnn_lr=config['koopman_nonnn_lr']
            )
    # sampler = RaySampler(agents=policy,
    #                 envs=env,
    #                 max_episode_length=env.spec.max_episode_length,
    #                 worker_class=DefaultWorker)
    sampler = LocalSampler(agents=policy,
                    envs=env,
                    max_episode_length=env.spec.max_episode_length,
                    worker_class=DefaultWorker)
                    
    if policy_type == 'vanilla':
        sac = SAC(env_spec=env.spec,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            sampler=sampler,
            gradient_steps_per_itr=1000,
            max_episode_length_eval=config['max_episode_length_eval'],
            replay_buffer=replay_buffer,
            min_buffer_size=1e4,
            target_update_tau=5e-3,
            discount=0.99,
            buffer_batch_size=256,
            reward_scale=1.,
            steps_per_epoch=1,
            num_evaluation_episodes=num_evaluation_episodes,
            )
    else:
        sac = KoopmanLQRSAC(env_spec=env.spec,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            sampler=sampler,
            gradient_steps_per_itr=1000,
            max_episode_length_eval=config['max_episode_length_eval'],
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
    # so far garage does not support ray sampler for gpu
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=config['num_epochs'], batch_size=config['batch_size'], plot=False)
    return

@wrap_experiment
def koopmanlqr_ppo_bullet_tests(ctxt=None, config=None):
    assert(config is not None)
    seed = config['seed']
    policy_type = config['policy_type']
    policy_horizon = config['koopman_horizon']

    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    env = normalize(BulletEnv(config['env_name']))

    # print(env.spec.observation_space)
    # print(env.spec.observation_space.flat_dim)


    #need a separate seed for gym environment for full determinism
    env.seed(seed)
    env.action_space.seed(seed)
    #original hidden size 256
    hidden_size = config['hidden_size']

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

        #need a separate hiddenzie for MLP because of the experience of using linearly parameterized approximator
        if config['valuefunc_type'] == 'koopman':
            #create a separate koopman control from a koopman_policy
            koopman = GaussianKoopmanLQRPolicy(
                    env_spec=env.spec,
                    k=4,
                    T=5,
                    phi=[hidden_size, hidden_size],
                    residual=None,
                    init_std=1.0,
                    use_state_goal='latent'
                )
            kpm_ctrl = koopman._kpm_ctrl
        
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
            k=config['koopman_size'],
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
                least_square_fit_coeff=config['least_square_fit_coeff'],    #use least square
                koopman_fit_coeff=config['koopman_fit_coeff'],
                koopman_fit_coeff_errbound=config['koopman_fit_coeff_errbound'],
                koopman_fit_optim_lr=config['koopman_fit_optim_lr'],
                koopman_fit_n_itrs=config['koopman_fit_n_itrs'],
                koopman_fit_mat_reg_coeff=config['koopman_fit_mat_reg_coeff'],
                koopman_recons_coeff=config['koopman_recons_coeff'],
                koopman_nonnn_lr=config['koopman_nonnn_lr']
            )

        # koopman_param._koopman_recons_coeff = -1
        if config['valuefunc_type'] == 'koopman':
            #use the same or should be a separaed one as sac?
            #i feel this might be to reusable because it gives chance to improve policy in value learning as well
            kpm_ctrl = policy._kpm_ctrl

    #shared settings
    if config['valuefunc_type'] == 'vanilla':
        print('Using Vanilla Function')    
        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                hidden_sizes=(hidden_size, hidden_size),
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)
    else:
        print('Using Koopman Value Function')
        value_function = GaussianKoopmanMLPValueFunction(kpm_ctrl=kpm_ctrl, #shared or another kpm lqr
                                                env_spec=env.spec,
                                                hidden_sizes=(hidden_size, hidden_size),
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)

    # sampler = MultiprocessingSampler(agents=policy,
    #                     envs=env,
    #                     max_episode_length=env.spec.max_episode_length,
    #                     worker_class=DefaultWorker)
    sampler = LocalSampler(agents=policy,
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
    trainer.train(n_epochs=config['num_epochs'], batch_size=config['batch_size'], plot=False)   
    return

import itertools
import os
import argparse
import yaml
import time
import wandb

CONFIG_PATH = './config'

def main(args):
    #load yaml configuration
    with open(os.path.join(CONFIG_PATH, args.config+'.yaml')) as file:
        config = yaml.safe_load(file)

    # seeds = [1, 21, 52, 251, 521]
    # seeds = [21, 52, 251, 521]
    # seeds = [251, 521]
    # seeds = [2, 12, 51, 125, 512]

    seeds = [1]
    policy_types = ['vanilla', 'koopman', 'koopman_residual']
    policy_types = ['koopman', 'koopman_residual']
    # policy_types = ['vanilla']

    valfunc_types = ['vanilla']
    wandb_tensorboard_patched = False

    for seed in seeds:
        config['seed'] = seed 
        for policy_type, valfunc_type in itertools.product(policy_types, valfunc_types):
            #update config params
            config['policy_type'] = policy_type
            config['valuefunc_type'] = valfunc_type
            
            #build experiment name
            timestr = time.strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join('data/local/experiment', '{0}_{1}_{2}_seed{3}_{4}'.format(config['rl_algo'], args.config, policy_type, seed, timestr))
            
            config.update(vars(args))
            if args.use_wandb:
                wandb_run = wandb.init(config=config, project='koopman', name=log_dir, reinit=True, sync_tensorboard=False)
                if not wandb_tensorboard_patched:
                    wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
                    wandb_tensorboard_patched = True

            ctxt = dict(log_dir=log_dir, snapshot_mode='last', archive_launch_repo=False, use_existing_dir=True)       

            if config['rl_algo'] == 'sac':
                koopmanlqr_sac_bullet_tests(ctxt, config=config)
            elif config['rl_algo'] == 'ppo':
                koopmanlqr_ppo_bullet_tests(ctxt, config=config)
            else:
                print('Unsupported RL algorithm.')
            
            if args.use_wandb:
                wandb_run.finish()

RIGID_EXP_NAMES = ['InvertedPendulumSwingup', 'InvertedPendulum', 'Block2D', 'HalfCheetah', 'Ant']
DEFORM_EXP_NAMES = ['HangGarment', 'Button']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args', add_help=False)
    parser.add_argument('--config', type=str,
                        default='InvertedPendulumSwingup', help='Name of the config file', choices=RIGID_EXP_NAMES+DEFORM_EXP_NAMES)
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to enable logging to wandb.ai')
    args, unknown = parser.parse_known_args()
    main(args)
