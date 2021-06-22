"""
Solving LQR for Koopman embedded dynamical systems. Keep everything differentiable.
"""
import numpy as np
import copy

import torch
from torch import nn, optim

from koopman_policy.utils import FCNN, batch_mv, soft_update_model

class KoopmanLQR(nn.Module):
    def __init__(self, k, x_dim, u_dim, x_goal, T, phi=None, u_affine=None, g_goal=None):
        """
        k:          rank of approximated koopman operator
        x_dim:      dimension of system state
        u_dim:      dimension of control input
        x_goal:     goal state
        T:          length of horizon
        phi:        observation func for state. nn.Module. should be a module handling input dim x_dim and output k dim vectors
                    use a linear subspace projection if kept None
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        g_goal:     None by default. If not, override the x_goal so it is not necessarily corresponding to a concrete goal state
                    might be useful for non regularization tasks.  
        """
        super().__init__()
        self._k = k
        self._x_dim = x_dim
        self._u_dim = u_dim
        if g_goal is None:
            if x_goal is not None:
                if isinstance(x_goal, nn.Module):
                    raise NotImplementedError
                else:
                    self._x_goal = nn.Parameter(x_goal)
            else:
                self._x_goal = x_goal
            self._g_goal = g_goal
        else:
            self._x_goal = None
            self._g_goal = nn.Parameter(g_goal)

        self._T = T

        # Koopman observable function: putting state and control together as an augmented autonomous state
        # and assume control is not subject a dynamical process h(x_t+1, 0) = K h(x_t, u_t)
        # only state part is nonlinearly embedded: h(x_t, u_t) = phi(x_t) + Cu_t
        # put together: h(x_t+1, 0) = A phi(x_t) + Bu_t --> phi(x_t+1) = A phi(x_t) + B u_t, this makes a linear system w.r.t. g(.)
        # also prepare a decoder for the inverse of lifting operation, this might be needed to shape the embedding space
        # add switches in fit function so one can extend that by using invertible transformation?
        if phi is None:
            self._phi = nn.Linear(x_dim, k)
            self._phi_inv = nn.Linear(k, x_dim)
        elif phi == 'FCNN':
            #use a simple fully connected neural network
            self._phi = FCNN(x_dim, k, 16)
            self._phi_inv = FCNN(k, x_dim, 16)
        elif isinstance(phi, list):
            self._phi = FCNN(x_dim, k, hidden_dim=phi)
            self._phi_inv = FCNN(k, x_dim, hidden_dim=list(reversed(phi)))
        else:
            self._phi = phi
            self._phi_inv = None # this, user must specify their own inverse network

        #retain a target phi to emulate q function learning
        self._target_phi = copy.deepcopy(self._phi)
        
        #prepare linear system params
        #self._phi_affine = nn.Linear(k, k, bias=False)
        self._phi_affine = nn.Parameter(torch.empty((k, k)))
        
        if u_affine is None:
            #self._u_affine = nn.Linear(u_dim, k, bias=False)
            self._u_affine = nn.Parameter(torch.empty((k, u_dim)))
        else:
            self._u_affine = nn.Parameter(u_affine)
        
        # try to avoid degenerated case, can it be fixed with initialization?
        torch.nn.init.normal_(self._phi_affine, mean=0, std=1.0)
        torch.nn.init.normal_(self._u_affine, mean=0, std=1.0)

        #parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(k))  #to use: Q = diag(_q_diag_log.exp())
        # self._q_diag_log.requires_grad = False
        #gain of control penalty, in theory need to be parameterized...
        self._r_diag_log = nn.Parameter(torch.zeros(u_dim))
        self._r_diag_log.requires_grad = False

        #zero tensor constant for k and v in the case of fixed origin
        #these will be automatically moved to gpu so no need to create and check in the forward process
        self.register_buffer('_zero_tensor_constant_k', torch.zeros((1, self._u_dim)))
        self.register_buffer('_zero_tensor_constant_v', torch.zeros((1, self._k)))

        #we may need to create a few cache for K, k, V and v because they are not dependent on x
        #unless we make g_goal depend on it. This allows to avoid repeatively calculate riccati recursion in eval mode
        self._riccati_solution_cache = None
        return
    
    def _solve_lqr(self, A, B, Q, R, goals):
        # a differentiable process of solving LQR, 
        # time-invariant A, B, Q, R (with leading batch dimensions) but goals can be a batch of trajectories (batch_size, T+1, k)
        #       min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        # s.t.  x_{t+1} = A x_t + B u_t
        # return feedback gain and feedforward terms such that u = -K x + k

        # goals include the terminal reference
        T = self._T

        K = [None] * T
        k = [None] * T

        V = [None] * (T+1)
        v = [None] * (T+1)

        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)
        #initialization for backpropagation
        V[-1] = Q
        if goals is not None:
            v[-1] = batch_mv(Q, goals[:, -1, :])
            for i in reversed(range(T)):
                # (B^T V B + R)^{-1}
                # V_uu_inv = torch.inverse(
                #     torch.matmul(
                #     torch.matmul(B_trans, V[i+1]),
                #     B
                #     ) + R
                # ) 
                # (B^T V B + R)^{-1} B^T
                # using torch.solve(B, A) to obtain the solution of A X = B to avoid direct inverse, note it also returns LU
                V_uu_inv_B_trans, _ = torch.solve(B_trans,
                    torch.matmul(torch.matmul(B_trans, V[i+1]),
                    B
                    ) + R)
                # V_uu_inv_B_trans = torch.matmul(
                #     torch.inverse(
                #     torch.matmul(
                #     torch.matmul(B_trans, V[i+1]),
                #     B
                #     ) + R
                #     ), B_trans 
                # )
                # K = (B^T V B + R)^{-1} B^T V A 
                K[i] = torch.matmul(
                        torch.matmul(
                            V_uu_inv_B_trans,
                            V[i+1]
                        ),
                        A
                    )

                k[i] = batch_mv(V_uu_inv_B_trans, v[i+1])

                #riccati difference equation
                # A-BK
                A_BK = A - torch.matmul(B, K[i])
                # V = A^T V (A-BK) + Q = A^T V A - A^T V B (B^T V B + R)^{-1} B^T V A + Q
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
                # v = (A-BK)^Tv + Q r
                v[i] = batch_mv(A_BK.transpose(-2, -1), v[i+1]) + batch_mv(Q, goals[:, i, :])
        else:
            #None goals means a fixed regulation point at origin. ignore k and v for efficiency
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of A X = B to avoid direct inverse, note it also returns LU
                V_uu_inv_B_trans, _ = torch.solve(B_trans,
                    torch.matmul(torch.matmul(B_trans, V[i+1]),
                    B
                    ) + R)
                # V_uu_inv_B_trans = torch.matmul(
                #     torch.inverse(
                #     torch.matmul(
                #     torch.matmul(B_trans, V[i+1]),
                #     B
                #     ) + R
                #     ), B_trans 
                # )
                # K = (B^T V B + R)^{-1} B^T V A 
                K[i] = torch.matmul(
                        torch.matmul(
                            V_uu_inv_B_trans,
                            V[i+1]
                        ),
                        A
                    )
                #riccati difference equation
                # A-BK
                A_BK = A - torch.matmul(B, K[i])
                # V = A^T V (A-BK) + Q = A^T V A - A^T V B (B^T V B + R)^{-1} B^T V A + Q
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
            k[:] = self._zero_tensor_constant_k
            v[:] = self._zero_tensor_constant_v       

        # we might need to cat or stack to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k, V, v

    def fit_koopman(self, X, U,         
            train_phi=True,         #whether train encoder or not
            train_phi_inv=True,     #include reconstruction aux loss
            train_metric=True,      #include auxiliary term to preserve distance
            ls_factor=None,         #if not None but a float, use pinv directly solving the least square problem, with the value as the regularization factor of matrix inversion
            n_itrs=100,             #number of optimization iterations
            lr=1e-4,                #learning rate
            verbose=False):         #verbose info
        '''
        fit koopman paramters, phi, A and B
        X, U:   trajectories of states and actions (batch_size, T, dim), self-supervision by ignoring the last action
        '''
        u_curr = U[:, :X.shape[1]-1, :]

        #choose to train NN feature or not. In the case of False, the embedding is basically a random projection
        if train_phi:
            # params = list(self._phi.parameters()) + list(self._phi_affine.parameters()) + list(self._u_affine.parameters())
            params = list(self._phi.parameters())
        if ls_factor is None:
            #this can actually solved by pinverse...
            # params += list(self._phi_affine.parameters()) + list(self._u_affine.parameters())
            params += [self._phi_affine, self._u_affine]

        if train_phi_inv:
            assert self._phi_inv is not None
            params += list(self._phi_inv.parameters())
        
        if train_phi or train_phi_inv or train_metric:
            #numeric optimization for all params
            opt = optim.Adam(params, lr=lr, weight_decay=0)
            loss = nn.MSELoss()

            # use pinv for least square for a differentiable system identification?
            # or lets stick to an entire optimization loop since we want to learn encoder/decoder anyhow
            for i in range(n_itrs):
                g = self._phi(X)
                if ls_factor is None:
                    g_next = g[:, 1:, :]
                    g_curr = g[:, :-1, :]
                    
                    g_pred = torch.matmul(g_curr, self._phi_affine.transpose(0, 1))+torch.matmul(u_curr, self._u_affine.transpose(0, 1))
                
                    #eval loss over all of them
                    tol_loss = loss(g_pred, g_next)
                    pred_loss = tol_loss.item()
                else:
                    #solve least square problem to get A and B
                    A, B, pred_loss = self._solve_least_square_traj(g, u_curr, ls_factor)
                    tol_loss = pred_loss
                    self._phi_affine = nn.Parameter(A)
                    self._u_affine = nn.Parameter(B)
                    
                if train_phi_inv:
                    recons_loss = loss(self._phi_inv(self._phi(X)), X)
                    tol_loss = tol_loss + recons_loss
                else:
                    recons_loss = -1
                
                if train_metric:
                    #using auxiliary cost to preserve distance in original and embedded space
                    metric_loss = self._loss_metric(X, g)
                    tol_loss = tol_loss + metric_loss * 0.5 #default weight...
                else:
                    metric_loss = -1

                #apply gradient
                opt.zero_grad()
                tol_loss.backward()
                opt.step()

                if verbose:
                    print('Iteration {0}: Pred loss - {1}; Recons loss - {2}; Metric loss - {3}'.format(i, pred_loss, recons_loss, metric_loss))
        return
    
    def _solve_least_square_traj(self, G, U, I_factor=10):
        '''
        using pinv to solve least square to identify A and B, do we need to retain batch dimension? Shouldn't we account all data?
        G, U: (B, T, dim)
        G_t+1 = G_t A^T +  u B^T = [G_t u] [A^T; B^T]
        '''
        B, T, dim = G.size()
        g_next_flatten = G[:, 1:, :].reshape(torch.Size([1, B*(T-1), dim]))
        g_curr_flatten = G[:, :-1, :].reshape(torch.Size([1, B*(T-1), dim]))
        u_flatten = U[:, :T-1, :].reshape(torch.Size([1, B*(T-1), U.shape[2]]))

        #assign that to affines
        return self._solve_least_square(g_curr_flatten, g_next_flatten, u_flatten, I_factor)
    
    def _solve_least_square(self, G, G_next, U, I_factor=10):
        '''
        G, G_next, U: flattened encodings, encodings of the next step and control - size(1, N, dim)
        G_next = G A^T + u B^T

        return mat (dim*dim), remove batch dimension by squeeze(0)
        '''
        #(1, B*T, k+u_dim)
        GU_cat = torch.cat([G, U], dim=2)
        AB_cat = torch.bmm(batch_pinv(GU_cat, I_factor, use_gpu=next(self.parameters()).is_cuda), G_next)

        A_transpose = AB_cat[:, :G.shape[2], :]
        B_transpose = AB_cat[:, G.shape[2]:, :]        
        # print(A_transpose, B_transpose)
        # print(A_transpose.squeeze(0), B_transpose.squeeze(0))
        fit_err = G_next - torch.bmm(GU_cat, AB_cat)
        fit_err = torch.sqrt((fit_err ** 2).mean())
        return A_transpose.squeeze(0).transpose(0,1), B_transpose.squeeze(0).transpose(0,1), fit_err
    
    def _loss_metric(self, X, G, scaling_factor = 1):
        #constructing auxiliary cost to preserve distance in original and embedded space
        #see Yunzhu et al, ICLR 2020

        tol_state = X.shape[0]*X.shape[1]
        permu = np.random.permutation(tol_state)
        split_0 = permu[:tol_state // 2]
        split_1 = permu[tol_state // 2:]

        X_flatview = X.view(torch.Size([X.size(0) * X.size(1)]) + X.size()[2:])
        G_flatview = G.view(torch.Size([G.size(0) * G.size(1)]) + G.size()[2:])
        
        dist_g = torch.mean((G_flatview[split_0] - G_flatview[split_1]) ** 2) #distance in the embedded/lifted space
        dist_s = torch.mean((X_flatview[split_0] - X_flatview[split_1]) ** 2) #distance in the original/state space
        #using scaling factor to compensate what? difference in dimension?
        loss_metric = torch.abs(dist_g * scaling_factor - dist_s).mean()
        return loss_metric

    def predict_koopman(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        return torch.matmul(G, self._phi_affine.transpose(0, 1))+torch.matmul(U, self._u_affine.transpose(0, 1))
    
    def _retrieve_riccati_solution(self):
        if self.training or self._riccati_solution_cache is None:
            Q = torch.diag(self._q_diag_log.exp()).unsqueeze(0)
            R = torch.diag(self._r_diag_log.exp()).unsqueeze(0)
            if self._x_goal is not None:
                #this might have efficiency issue since the goal needs to be populated every call?
                goals = torch.repeat_interleave(self._phi(self._x_goal).unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
            else:
                #use g_goal instead
                if self._g_goal is not None:
                    goals = torch.repeat_interleave(self._g_goal.unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
                else:
                    goals = None

            K, k, V, v = self._solve_lqr(self._phi_affine.unsqueeze(0), self._u_affine.unsqueeze(0), Q, R, goals)
            self._riccati_solution_cache = (
                [tmp.detach().clone() for tmp in K], 
                [tmp.detach().clone()  for tmp in k], 
                [tmp.detach().clone()  for tmp in V], 
                [tmp.detach().clone()  for tmp in v])
        else:
            K, k, V, v = self._riccati_solution_cache
        return K, k, V, v

    def forward(self, x0):
        '''
        perform mpc with current parameters given the initial x0
        '''
        K, k, V, v = self._retrieve_riccati_solution()
        #apply the first control as mpc
        # print(K[0].shape, k[0].shape)
        u = -batch_mv(K[0], self._phi(x0)) + k[0] 
        return u
    
    def forward_quadratic_cost(self, x, u=None):
        #evaluate the cost of x with quadratic parameters
        #x:         (B, d_x)
        #u:         (B, d_u)
        #return :   (B,)
        if isinstance(self._x_goal, nn.Module):
            raise NotImplementedError
        else:
            if self._x_goal is not None:
                #populate g_goal with x_goal
                g_goal = self._phi(self._x_goal) 
            else:
                g_goal = self._g_goal

            cost = torch.sum((self._phi(x)-g_goal)**2 * self._q_diag_log.exp()[None, :], dim=1)

            if u is not None:
                cost = cost + torch.sum(u**2 * self._r_diag_log.exp()[None, :], dim=1)
        return cost
    
    def forward_cost_to_go(self, x0):
        #evaluate the cost-to-go of x up to a constant
        #x:         (B, d_x)
        #return:    (B,)
        K, k, V, v = self._retrieve_riccati_solution()

        phi = self._phi(x0)
        cost_to_go = (phi * batch_mv(V[0], phi)).sum(-1) - 2*(phi * v[0]).sum(-1)
        return cost_to_go
    
    def _koopman_fit_loss(self, x, x_next, u, ls_factor, target_tau=-1):
        g = self._phi(x)

        if target_tau > 0:
            g_next = self._target_phi(x_next)
        else:
            g_next = self._phi(x_next)

        if ls_factor > 0:
            A, B, fit_err = self._solve_least_square(g.unsqueeze(0), g_next.unsqueeze(0), u.unsqueeze(0), I_factor=ls_factor)
            #assign A and B to control parameter for future evaluation
            self._phi_affine = nn.Parameter(A, requires_grad=False)
            self._u_affine = nn.Parameter(B, requires_grad=False)
        else:
            g_pred = torch.matmul(g, self._phi_affine.transpose(0, 1))+torch.matmul(u, self._u_affine.transpose(0, 1))
            loss = nn.MSELoss()
            fit_err = loss(g_pred, g_next)   
        
        return fit_err
    
    def _koopman_recons_loss(self, x):
        # assert self._phi_inv is not None
        g = self._phi(x)
        x_recons = self._phi_inv(g)

        loss = nn.MSELoss()
        recons_err = loss(x, x_recons)
        return recons_err
    
    def _koopman_matreg_loss(self):
        matreg_loss = torch.norm(self._phi_affine, p=2) + torch.norm(self._u_affine, p=2)
        return matreg_loss
    
    def _update_target_phi(self, tau):
        soft_update_model(self._target_phi, self._phi, tau)
        return