import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal

from garage.torch.policies.stochastic_policy import StochasticPolicy

from garage.torch import global_device

import koopman_policy.koopman_lqr as kpm

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
        use_state_goal (str/Tensor): whether to learn a goal in the state space or not. 
                                    'state'/'latent': the goal will be learned in the state/latent space.
                                    otherwise, it will assume a fixed origin in the latent space
                                    And if a Tensor is given, treat it as a regularization task with a known state goal. 
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
                use_state_goal='fixed_origin',
                name='GaussianKoopmanLQRPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._use_state_goal = use_state_goal
        if isinstance(use_state_goal, str):
            x_goal, g_goal = None, None
            if use_state_goal == 'state':
                x_goal=torch.zeros(self._obs_dim) #set x_goal separately if we know the goal
            elif use_state_goal == 'latent':
                g_goal=torch.zeros(k) #set x_goal separately if we know the goal
            else:
                pass #no goal is specified, assume a fixed origin in the latent space
            self._kpm_ctrl = kpm.KoopmanLQR(k=k, x_dim=self._obs_dim, u_dim=self._action_dim, 
                                x_goal=x_goal, T=T, phi=phi, u_affine=None, g_goal=g_goal) 
        else:
            #regularization for a known goal
            self._kpm_ctrl = kpm.KoopmanLQR(k=k, x_dim=self._obs_dim, u_dim=self._action_dim, 
                                x_goal=use_state_goal, T=T, phi=phi, u_affine=None, g_goal=None) #set x_goal separately if we know the goal
            self._kpm_ctrl._x_goal.requires_grad = False

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
    
    def set_state_goal_learnable(self, state_goal=None, learnable=True):
        if self._kpm_ctrl._x_goal is not None:
            if state_goal is not None:
                assert(state_goal.shape == self._kpm_ctrl._x_goal.shape)
                self._kpm_ctrl._x_goal = nn.Parameter(state_goal)
            self._kpm_ctrl._x_goal.requires_grad=learnable
        else:
            print('No state goal is used.')
        return
    
    def get_koopman_params(self):
        param = list(self._kpm_ctrl._phi.parameters())
        if self._kpm_ctrl._phi_inv is not None:
            param+=list(self._kpm_ctrl._phi_inv.parameters())
        return param
    
    def get_lindyn_params(self):
        return [self._kpm_ctrl._phi_affine, self._kpm_ctrl._u_affine]
    
    def get_qr_params(self):
        param = [self._kpm_ctrl._q_diag_log, self._kpm_ctrl._r_diag_log]
        if self._kpm_ctrl._x_goal is not None:
            param.append(self._kpm_ctrl._x_goal)
        if self._kpm_ctrl._g_goal is not None:
            param.append(self._kpm_ctrl._g_goal)
        return param


class KoopmanLQRRLParam():
    def __init__(
            self,
            #a separate learning rate for non NN parameters, maybe we need a different learning rate for them?
            koopman_nonnn_lr=None,  
            #regularization term for least square if >0. -1 for not using least square to fit koopman
            least_square_fit_coeff=-1, 
            #weight to account for koopman fit error, -1 means not to account it
            #this will merge koopman objectives with the main objective and apply gradient all together
            koopman_fit_coeff=-1,
            koopman_fit_coeff_errbound=-1,  #optimize koopman fit coefficient as lagrangian multipler as well to enforce the constraint of fit_err <= errbound
            #otherwise, can also use a separate optimizer for alternating gradient descent, this will overlap the above settings
            koopman_fit_optim_lr=-1,        #learning rate for the koopman fit optimizer
            koopman_fit_n_itrs=1,           #number of iterations for a separate
            koopman_fit_mat_reg_coeff=1e-3,    #coefficient to penalize the norm of A and B
            #weight to account for reconstruction error from koopman observables, -1 means to ignore the term
            #shall we also have a separate optimizer for reconstruction? now lets stick to the same one with a different weight if this is needed
            koopman_recons_coeff=-1,
            koopman_target_update_tau_phi=-1,
            ):
        self._koopman_nonnn_lr = koopman_nonnn_lr
        self._least_square_fit_coeff = least_square_fit_coeff
        self._koopman_fit_coeff = koopman_fit_coeff
        self._koopman_fit_coeff_errbound = koopman_fit_coeff_errbound
        self._koopman_fit_optim_lr = koopman_fit_optim_lr
        self._koopman_fit_n_itrs = koopman_fit_n_itrs
        self._koopman_fit_mat_reg_coeff = koopman_fit_mat_reg_coeff
        self._koopman_recons_coeff = koopman_recons_coeff
        self._koopman_target_update_tau_phi = koopman_target_update_tau_phi
        return
    
from garage.torch.value_functions import GaussianMLPValueFunction
class GaussianKoopmanMLPValueFunction(GaussianMLPValueFunction):
    def __init__(self, kpm_ctrl, **kwargs) -> None:
        super().__init__(**kwargs)
        
        #not to build it from the scratch for possible sharing the same koopman system with the policy
        self._kpm_ctrl = kpm_ctrl
        return

    def forward_koopman(self, obs):
        #return value func/negative cost-to-go associated to the koopman system
        return -self._kpm_ctrl.forward_cost_to_go(obs)
    
    def forward(self, obs):
        nn_val = super().forward(obs)
        koopman_val = self.forward_koopman(obs)
        return nn_val + koopman_val

from garage.torch.q_functions import ContinuousMLPQFunction
class ContinuousKoopmanMLPQFunction(ContinuousMLPQFunction):
    def __init__(self, kpm_ctrl, **kwargs) -> None:
        super().__init__(**kwargs)
        self._kpm_ctrl = kpm_ctrl
        return
    
    def forward_koopman(self, obs, act):
        return -self._kpm_ctrl.forward_cost_ctrl_to_go(obs, act)
    
    def forward(self, observations, actions):
        nn_val = super().forward(observations, actions)
        koopman_val = self.forward_koopman(observations, actions)
        #note this is different from value func that will be called by ppo and fed (B, T, dx) and returned vals are (B, T)
        #here nn_val will return (B, 1) while koopman returns (B,)
        return nn_val + koopman_val.unsqueeze(-1)
