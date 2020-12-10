import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal

from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.policies.policy import Policy

from garage.torch import global_device

import koopman_lqr as kpm

from dowel import logger, tabular

class GaussianKoopmanLQRPolicy(StochasticPolicy):
    """MLP whose outputs are fed into a Normal distribution..

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        k:          dimension of reduced koopman embedding
        T:          horizon of koopman mpc
        phi:        nn.Module to transform observation to koopman embeddings, need to be compatible with k and _obs_dim
        residual:   nn.Module to consider the residual with koopman mpc as feedforward. Only use koopman mpc if None
        normal_distribution_cls (torch.distribution): normal distribution class
        to be constructed and returned by a call to forward. By default, is
        `torch.distributions.Normal`.
        init_std (number): initial standard deviation parameter
        name (str): Name of policy.

    """
    def __init__(self,
                env_spec,
                k=3,
                T=5,
                phi='FCNN',
                residual=None,
                normal_distribution_cls=Normal,
                init_std=1.0,
                name='GaussianKoopmanLQRPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._kpm_ctrl = kpm.KoopmanLQR(k=k, x_dim=self._obs_dim, u_dim=self._action_dim, 
                        x_goal=torch.zeros(self._obs_dim), T=T, phi=phi, u_affine=None) #set x_goal separately if we know the goal
        
        self._residual = residual
        self._normal_distribution_cls=normal_distribution_cls
        #this is probably slightly different from GaussianMLP that has only one param for variance
        init_std_param = torch.Tensor([init_std]).log()
        self._init_std = torch.nn.Parameter(init_std_param)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        # logger.log('Obervations shape: {0}, {1}'.format(observations.shape[0], observations.shape[1]))
        #first flatten observations because jacobian_in_batch can only handle one batch dimension
        #should we use view to avoid create new tensors?
        obs_flatten = torch.reshape(observations, (-1, self._obs_dim))
        # logger.log('Obervations flatten shape: {0}, {1}'.format(obs_flatten.shape[0], obs_flatten.shape[1]))
        #might need to figure out a way for more axes
        mean_flatten = self._kpm_ctrl(obs_flatten)

        if self._residual is not None:
            mean_flatten = mean_flatten + self._residual(obs_flatten)

        #restore mean shape
        broadcast_shape = list(observations.shape[:-1]) + [self._action_dim]
        mean = torch.reshape(mean_flatten, broadcast_shape)

        uncentered_log_std = torch.zeros(*broadcast_shape).to(
                    global_device()) + self._init_std

        std = uncentered_log_std.exp()

        dist = self._normal_distribution_cls(mean, std)

        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))