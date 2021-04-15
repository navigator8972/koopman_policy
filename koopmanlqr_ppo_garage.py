##
## wrapper of garage PPO with Koopman auxiliary objectives
import numpy as np

import torch
import torch.nn as nn

from dowel import tabular

from garage.torch.algos import PPO
from garage import log_performance, obtain_evaluation_episodes, StepType

from garage.torch import np_to_torch, torch_to_np, dict_np_to_torch

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
        return
    
    def _koopman_fit_objective(self, samples_data):
        obs = samples_data['observation']
        acts = samples_data['action']
        # rewards = samples_data['reward'].flatten()
        # terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']

        fit_err = self.policy._kpm_ctrl._koopman_fit_loss(obs, next_obs, acts, self._koopman_param._least_square_fit_coeff)

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
            tol_obj = tol_obj


        return tol_obj
