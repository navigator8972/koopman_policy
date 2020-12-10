import torch
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from koopmanlqr_policy_garage import GaussianKoopmanLQRPolicy 

@wrap_experiment
def ppo_half_cheetah(ctxt=None, seed=1):
    """Train PPO with HalfCheetah-2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    env = GymEnv(gym.make('HalfCheetah-v2'))

    trainer = Trainer(ctxt)

    policy = GaussianKoopmanLQRPolicy(
        env_spec=env.spec,
        k=4,
        T=5,
        phi='FCNN',
        residual=None,
        # normal_distribution_cls=Normal,
        init_std=1.0,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=1500, plot=True)


ppo_half_cheetah(seed=1)