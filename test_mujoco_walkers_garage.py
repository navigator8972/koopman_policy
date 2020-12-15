import torch
import torch.nn as nn
import gym

from garage import wrap_experiment
from garage.envs import GymEnv

from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler

from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
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
    env = GymEnv(gym.make('HalfCheetah-v2'))

    trainer = Trainer(ctxt)

    # policy = GaussianMLPPolicy(env.spec,
    #                            hidden_sizes=[64, 64],
    #                            hidden_nonlinearity=torch.tanh,
    #                            output_nonlinearity=None)

    in_dim = env.spec.observation_space.flat_dim
    hidden_dim = 32
    out_dim = env.spec.action_space.flat_dim

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
        T=5,
        phi='FCNN',
        residual=residual,
        # normal_distribution_cls=Normal,
        init_std=1.0,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
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
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1500, plot=True)


ppo_mujoco_walkers(seed=1)