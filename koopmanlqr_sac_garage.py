import numpy as np

import torch
import torch.nn as nn

from dowel import tabular

from garage.torch.algos import SAC
from garage import log_performance, obtain_evaluation_episodes, StepType

from garage.torch import np_to_torch, torch_to_np, dict_np_to_torch

from koopman_policy.koopmanlqr_policy_garage import KoopmanLQRRLParam

class KoopmanLQRSAC(SAC):
    def __init__(self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            #additional params for koopman policy
            koopman_param=KoopmanLQRRLParam()
            ):
        super().__init__(env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            max_episode_length_eval=max_episode_length_eval,
            gradient_steps_per_itr=gradient_steps_per_itr,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_episodes=num_evaluation_episodes,
            eval_env=eval_env,
            use_deterministic_evaluation=use_deterministic_evaluation
            )
        self._koopman_param = koopman_param

        if self._koopman_param._least_square_fit_coeff > 0:
            #the matrices will now be the result of least square procedure
            self.policy._kpm_ctrl._phi_affine.requires_grad = False
            self.policy._kpm_ctrl._u_affine.requires_grad = False
        
        if self._koopman_param._koopman_recons_coeff < 0:
            if self.policy._kpm_ctrl._phi_inv is not None:
                for param in self.policy._kpm_ctrl._phi_inv.parameters():
                    param.requires_grad = False
        
        if self._koopman_param._koopman_fit_optim_lr > 0 and self._koopman_param._koopman_fit_n_itrs > 0:
            #supress weights for merging objectives and override original policy to separate parameter learning
            #we may also let the policy optimizer learn everything but an initial test finds this might be a bad idea
            policy_param = self.policy.get_qr_params()
            if self.policy._residual is not None:
                policy_param += list(self.policy._residual.parameters())
            self._policy_optimizer = self._optimizer(policy_param, lr=policy_lr)
            self._koopman_aux_optimizer = self._optimizer(self.policy.get_koopman_params(),
                                                    lr=self._koopman_param._koopman_fit_optim_lr)
            if self._koopman_param._least_square_fit_coeff < 0:
                #add affine matrices as parameters as well
                self._koopman_aux_optimizer.param_groups.append({'params': self.policy._kpm_ctrl._phi_affine })
                self._koopman_aux_optimizer.param_groups.append({'params': self.policy._kpm_ctrl._u_affine })
        else:
            self._koopman_aux_optimizer = None
        return
    
    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.
            Statistics such as (average) discounted return and success rate are
            recorded.
        Args:
            epoch (int): The current training epoch.
        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes
        """
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)
        #add to log: latent var correlation, prediction acc and other indicators specialized for koopman_lqr
        self._log_koopmanlqr_statistics(epoch, eval_episodes, prefix='Evaluation')
        return last_return
    
    def _log_koopmanlqr_statistics(self, itr, batch, prefix='Evaluation'):
        # print(batch.observations.shape)
        obs = np_to_torch(batch.observations[:-1, :])
        next_obs = np_to_torch(batch.observations[1:, :])
        acts = np_to_torch(batch.actions[:-1, :])

        with torch.no_grad():
            g = self.policy._kpm_ctrl._phi(obs)
            g_next = self.policy._kpm_ctrl._phi(next_obs)

            g_pred = torch.matmul(g, self.policy._kpm_ctrl._phi_affine.transpose(0, 1))+torch.matmul(acts, self.policy._kpm_ctrl._u_affine.transpose(0, 1))
            loss = nn.MSELoss()
            fit_err = loss(g_pred, g_next)  

            #can only evaluate covariance for this
            #this might be a bad idea by taking collected rollouts as normally distributed.
            # print(self.policy._kpm_ctrl._k, self.env_spec.observation_space.flat_dim)
            # if self.policy._kpm_ctrl._k == self.env_spec.observation_space.flat_dim:
            #     corrcoef_det = np.linalg.det(np.corrcoef(batch.observations[1:, :], torch_to_np(g_pred)))
            if self._koopman_param._koopman_recons_coeff > 0:
                obs_recons = self.policy._kpm_ctrl._phi_inv(g)
                recons_err = loss(obs, obs_recons)

        with tabular.prefix(prefix + '/'):
            tabular.record('Koopman Fit Error', fit_err.item())
            # if self.policy._kpm_ctrl._k == self.env_spec.observation_space.flat_dim:
            #     tabular.record('Pearson Correlation', corrcoef_det)
            tabular.record('Koopman Fit Coeff', self._koopman_param._koopman_fit_coeff)
            
            if self._koopman_param._koopman_recons_coeff > 0:
                tabular.record('Koopman Recons Error', recons_err.item())
        return

    def _koopman_fit_objective(self, samples_data):
        obs = samples_data['observation']
        # !!!! This, needs to go through an inverted tanh?
        acts = samples_data['action']
        # rewards = samples_data['reward'].flatten()
        # terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']

        g = self.policy._kpm_ctrl._phi(obs)
        g_next = self.policy._kpm_ctrl._phi(next_obs)

        if self._koopman_param._least_square_fit_coeff > 0:
            A, B, fit_err = self.policy._kpm_ctrl._solve_least_square(g.unsqueeze(0), g_next.unsqueeze(0), acts.unsqueeze(0), I_factor=self._least_square_fit_coeff)
            #assign A and B to control parameter for future evaluation
            self.policy._kpm_ctrl._phi_affine = nn.Parameter(A, requires_grad=False)
            self.policy._kpm_ctrl._u_affine = nn.Parameter(B, requires_grad=False)
        else:
            g_pred = torch.matmul(g, self.policy._kpm_ctrl._phi_affine.transpose(0, 1))+torch.matmul(acts, self.policy._kpm_ctrl._u_affine.transpose(0, 1))
            loss = nn.MSELoss()
            fit_err = loss(g_pred, g_next)   

        return fit_err
    
    def _koopman_recons_objective(self, samples_data):
        obs = samples_data['observation']
        g = self.policy._kpm_ctrl._phi(obs)
        obs_recons = self.policy._kpm_ctrl._phi_inv(g)

        loss = nn.MSELoss()
        recons_err = loss(obs, obs_recons)
        return recons_err
    
    def optimize_koopman_aux(self):
        assert(self.optimize_koopman_aux is not None)

        for _ in range(self._koopman_param._koopman_fit_n_itrs):
            #note these are different from the policy training data
            if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
                samples = self.replay_buffer.sample_transitions(
                    self._buffer_batch_size)
                samples_data = dict_np_to_torch(samples)

            # fit err
            tol_loss = self._koopman_fit_objective(samples_data)

            #now bind recons term with koopman fit, because i dont see why to only use recons as the aux obj
            if self._koopman_param._koopman_recons_coeff > 0:
                koopman_recons_err = self._koopman_recons_objective(samples_data)
                tol_loss = tol_loss + self._koopman_param._koopman_recons_coeff * koopman_recons_err

            self._koopman_aux_optimizer.zero_grad()
            tol_loss.backward()

            self._koopman_aux_optimizer.step()
        return

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.
        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`
        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.
        """
        #standalone optimization loop for koopman aux
        if self._koopman_aux_optimizer is not None:
            self.optimize_koopman_aux()

        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        action_dists = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        
        if self._koopman_param._koopman_fit_coeff > 0 and self._koopman_aux_optimizer is None:
            koopman_fit_err = self._koopman_fit_objective(samples_data)
            tol_loss = policy_loss + self._koopman_param._koopman_fit_coeff * koopman_fit_err

            if self._koopman_param._koopman_fit_mat_reg_coeff > 0:
                tol_loss = tol_loss + self._koopman_param._koopman_fit_mat_reg_coeff * (
                    torch.norm(self.policy._kpm_ctrl._phi_affine, p=2) + torch.norm(self.policy._kpm_ctrl._u_affine, p=2))

            #now bind recons term with koopman fit, because i dont see why to only use recons as the aux obj
            if self._koopman_param._koopman_recons_coeff > 0:
                koopman_recons_err = self._koopman_recons_objective(samples_data)
                tol_loss = tol_loss + self._koopman_param._koopman_recons_coeff * koopman_recons_err
        else:
            tol_loss = policy_loss

        self._policy_optimizer.zero_grad()
        tol_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # for now disable this because it is unclear whether it helps to adapt for constraining a really small fit error 
        # if self._koopman_param._koopman_fit_coeff_errbound > 0:
        #     self._koopman_param._koopman_fit_coeff = max(0, 
        #             self._koopman_param._koopman_fit_coeff - self._policy_lr*(self._koopman_param._koopman_fit_coeff_errbound - koopman_fit_err.item()))

        return policy_loss, qf1_loss, qf2_loss
