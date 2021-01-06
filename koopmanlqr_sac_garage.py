import numpy as np
import torch
import torch.nn as nn

from garage.torch.algos import SAC
from garage import log_performance, obtain_evaluation_episodes, StepType

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
            use_least_square_fit=True
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
        self._use_least_square_fit=use_least_square_fit

        if use_least_square_fit:
            #the matrices will now be the result of least square procedure
            self.policy._kpm_ctrl._phi_affine.requires_grad = False
            self.policy._kpm_ctrl._u_affine.requires_grad = False
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
        self._log_koopmanlqr_statistics(epoch, eval_episodes, self._discount)
        return last_return
    
    def _log_koopmanlqr_statistics(self, itr, batch, discount, prefix='Evaluation'):
        #TODO
        return

    def _koopman_fit_objective(self, samples_data):
        obs = samples_data['observation']
        actions = samples_data['action']
        # rewards = samples_data['reward'].flatten()
        # terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        g = self.policy._kpm_ctrl._phi(obs)
        g_next = self.policy._kpm_ctrl._phi(next_obs)

        if self._use_least_square_fit:
            A, B, fit_err = self.policy._kpm_ctrl._solve_least_square(g.unsqueeze(0), g_next.unsqueeze(0), actions.unsqueeze(0), ls_factor=10)
            #assign A and B to control parameter for future evaluation
            self.policy._kpm_ctrl._phi_affine = nn.Parameter(A, requires_grad=False)
            self.policy._kpm_ctrl._u_affine = nn.Parameter(B, requires_grad=False)
        else:
            g_pred = torch.matmul(g, self.policy._kpm_ctrl._phi_affine.transpose(0, 1))+torch.matmul(actions, self._u_affine.transpose(0, 1))
            loss = nn.MSELoss()
            fit_err = loss(g_pred, g_next)   
        
        return fit_err
    
    def optimize_policy(self, samples_data):
        super().optimize_policy(samples_data)
        #TOADD: optimize for fit error as well - merge them with actor objective?

        return