#try cem on dedo tasks
#!/usr/bin/env python3
from garage import wrap_experiment
from garage.envs.bullet import BulletEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.algos import CEM
from garage.sampler import LocalSampler, MultiprocessingSampler
from garage.np.policies import FixedPolicy
from garage.trainer import Trainer

import numpy as np
import koopman_policy

class TimeIndexTrajectoryPolicy(FixedPolicy):
    def __init__(self, env_spec, T):
        #prepare a sequence of actions according to action spec
        action_seq = np.zeros((T ,env_spec.action_space.flat_dim))
        super().__init__(env_spec, action_seq)
    
    def set_param_values(self, params):
        self._scripted_actions = np.copy(params).reshape((-1, self._env_spec.action_space.flat_dim))
        return

    def get_param_values(self):
        return self._scripted_actions.flatten()

@wrap_experiment(snapshot_mode='last', archive_launch_repo=False)
def cem_dedo(ctxt=None, seed=1):
    """Train CEM
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    env = normalize(BulletEnv('HangBagBulletEnv-v0'))

    #a sequence of actions...
    policy = TimeIndexTrajectoryPolicy(env_spec=env.spec,
                                    T=env.spec.max_episode_length)

    n_samples = 20

    sampler = MultiprocessingSampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            )

    algo = CEM(env_spec=env.spec,
                policy=policy,
                sampler=sampler,
                best_frac=0.05,
                n_samples=n_samples)

    trainer.setup(algo, env)
    trainer.train(n_epochs=100, batch_size=env.spec.max_episode_length)


cem_dedo(seed=1)