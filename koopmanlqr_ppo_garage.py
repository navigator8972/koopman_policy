##
## wrapper of garage PPO with Koopman auxiliary objectives
import numpy as np

import torch
import torch.nn as nn

from dowel import tabular

from garage.torch.algos import PPO
from garage.torch import global_device

from koopman_policy.koopmanlqr_policy_garage import KoopmanLQRRLParam

class KoopmanLQRPPO(PPO):
    def __init__(self,
                env_spec,
                policy,
                value_function,
                sampler,
                policy_optimizer=None,
                vf_optimizer=None,
                lr_clip_range=2e-1,
                num_train_per_epoch=1,
                discount=0.99,
                gae_lambda=0.97,
                center_adv=True,
                positive_adv=False,
                policy_ent_coeff=0.0,
                use_softplus_entropy=False,
                stop_entropy_gradient=False,
                entropy_method='no_entropy',
                #extra parameter for koopman
                koopman_param=KoopmanLQRRLParam(),
                ):

        super().__init__(env_spec,
                policy,
                value_function,
                sampler,
                policy_optimizer,
                vf_optimizer,
                lr_clip_range,
                num_train_per_epoch,
                discount,
                gae_lambda,
                center_adv,
                positive_adv,
                policy_ent_coeff,
                use_softplus_entropy,
                stop_entropy_gradient,
                entropy_method)

        self._koopman_param = koopman_param
        self._policy_lr = 2.5e-4   #default of garage PPO
        nonnn_lr = koopman_param._koopman_nonnn_lr if koopman_param._koopman_nonnn_lr is not None else self._policy_lr
        #overload original policy optimizer if residual exists
        if self.policy._residual is not None:
            #apply weight decay to regularize residual part
            policy_optim_params = [{'params': self.policy.get_koopman_params()},
                {'params': self.policy.get_qr_params(), 'lr':nonnn_lr},
                {'params': self.policy._init_std},
                {'params': self.policy._residual.parameters(), 'weight_decay': 0.05}]
        else:
            policy_optim_params = [{'params': self.policy.get_koopman_params()},
                {'params': self.policy._init_std},      #dont forget about the exploration std of Gaussian policy
                {'params': self.policy.get_qr_params(), 'lr':nonnn_lr}]
        
        if self._koopman_param._least_square_fit_coeff < 0:
            policy_optim_params.append({'params':self.policy.get_lindyn_params(), 'lr':nonnn_lr})

        if policy_optimizer is None:
            policy_optimizer = torch.optim.Adam
            #note by default PPO/VPG will use a wrapper to construct the optimizer. we need to update the nested one
            self._policy_optimizer._optimizer = policy_optimizer(policy_optim_params, lr=self._policy_lr)
        else:
            self._policy_optimizer = policy_optimizer(policy_optim_params, lr=self._policy_lr)
        return
    
    def _koopman_fit_objective(self, samples_data):
        obs = samples_data['observation']
        acts = samples_data['action']
        # rewards = samples_data['reward'].flatten()
        # terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']

        target_tau = self._koopman_param._koopman_target_update_tau_phi
        fit_err = self.policy._kpm_ctrl._koopman_fit_loss(obs, next_obs, acts, 
            self._koopman_param._least_square_fit_coeff, target_tau)

        return fit_err
    
    def _koopman_recons_objective(self, samples_data):
        obs = samples_data['observation']
        recons_err = self.policy._kpm_ctrl._koopman_recons_loss(obs)
        return recons_err

    def _compute_objective(self, advantages, obs, actions, rewards):
        ppo_obj = super()._compute_objective(advantages, obs, actions, rewards)

        #augment koopman auxliary losses
        #note this is currently also optimized on policy but actually we can use off-policy data for the auxiliary part like UNREAL
        #we probably also need this by sampling separately from a replay buffer. that would need more extension
        samples_data = {
            'observation': obs[:-1, :],
            'action': actions[:-1, :],
            'next_observation': obs[1:, :]
        }
        #note PPO will maximize this objective so negate the sign of auxliary terms
        if self._koopman_param._koopman_fit_coeff > 0:
            koopman_fit_err = self._koopman_fit_objective(samples_data)
            tol_obj = ppo_obj - self._koopman_param._koopman_fit_coeff * koopman_fit_err

            if self._koopman_param._koopman_fit_mat_reg_coeff > 0:
                tol_obj = tol_obj - self._koopman_param._koopman_fit_mat_reg_coeff * self.policy._kpm_ctrl._koopman_matreg_loss()

            #now bind recons term with koopman fit, because i dont see why to only use recons as the aux obj
            if self._koopman_param._koopman_recons_coeff > 0:
                koopman_recons_err = self._koopman_recons_objective(samples_data)
                tol_obj = tol_obj - self._koopman_param._koopman_recons_coeff * koopman_recons_err
            
            with tabular.prefix('KoopmanAux/'):
                tabular.record('Koopman Fit Error', koopman_fit_err.item())
                # if self.policy._kpm_ctrl._k == self.env_spec.observation_space.flat_dim:
                #     tabular.record('Pearson Correlation', corrcoef_det)
                tabular.record('Koopman Fit Coeff', self._koopman_param._koopman_fit_coeff)
                
                if self._koopman_param._koopman_recons_coeff > 0:
                    tabular.record('Koopman Recons Error', koopman_recons_err.item())
        else:
            tol_obj = ppo_obj


        return tol_obj

    def _train_once(self, itr, eps):
        self.policy.train()
        res = super()._train_once(itr, eps)
        #update target phi
        if self._koopman_param._koopman_target_update_tau_phi > 0:
            self.policy._kpm_ctrl._update_target_phi(self._koopman_param._koopman_target_update_tau_phi)
        return res
    
    @property
    def networks(self):
        """Return all the networks within the model.
        Returns:
            list: A list of networks.
        """
        return [
            self.policy, self._value_function
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.
        Args:
            device (str): ID of GPU or CPU.
        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)

