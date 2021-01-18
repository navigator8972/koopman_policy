"""
Solving LQR for Koopman embedded dynamical systems. Keep everything differentiable.
"""
import numpy as np

import torch
from torch import nn, optim

import utils

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_nonlinearity=None, output_nonlinearity=None):
        super().__init__()
        if isinstance(hidden_dim, int): 
            self.network = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif isinstance(hidden_dim, list):
            modules = []
            modules_dim = [in_dim]+hidden_dim+[out_dim]
            for idx in range(len(modules_dim)-1):
                modules.append(nn.Linear(modules_dim[idx], modules_dim[idx+1]))
                if hidden_nonlinearity is None:
                    modules.append(nn.ReLU())
                else:
                    modules.append(hidden_nonlinearity())
            if output_nonlinearity is not None:
                modules.append(output_nonlinearity())

            self.network = nn.Sequential(*modules)
        else:
            # print(hidden_dim)
            raise NotImplementedError

        self.init_params()


    def forward(self, x):
        return self.network(x)
    
    def init_params(self):
        def param_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.apply(param_init)

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
            if isinstance(x_goal, nn.Module):
                raise NotImplementedError
            else:
                self._x_goal = nn.Parameter(x_goal)
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
        
        #prepare linear system params
        #self._phi_affine = nn.Linear(k, k, bias=False)
        self._phi_affine = nn.Parameter(torch.randn((k, k)))


        if u_affine is None:
            #self._u_affine = nn.Linear(u_dim, k, bias=False)
            self._u_affine = nn.Parameter(torch.randn((k, u_dim)))
        else:
            self._u_affine = nn.Parameter(u_affine)
        
        #parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(k))  #to use: Q = diag(_q_diag_log.exp())
        # self._q_diag_log.requires_grad = False
        #gain of control penalty, in theory need to be parameterized...
        self._r_diag_log = nn.Parameter(torch.zeros(u_dim))
        self._r_diag_log.requires_grad = False
        return
    
   
    def _solve_lqr(self, A, B, Q, R, goals):
        # a differentiable process of solving LQR, 
        # time-invariant A, B, Q, R (with leading batch dimensions) but goals can be a batch of trajectories (batch_size, T+1, k)
        #       min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        # s.t.  x_{t+1} = A x_t + B u_t
        # return feedback gain and feedforward terms such that u = -K x + k

        # goals include the terminal reference
        T = goals.shape[1]-1

        K = [None] * T
        k = [None] * T

        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)
        #initialization for backpropagation
        V = Q
        v = utils.batch_mv(Q, goals[:, -1, :])
        for i in reversed(range(T)):
            # (B^T V B + R)^{-1}
            V_uu_inv = torch.inverse(
                torch.matmul(
                torch.matmul(B_trans, V),
                B
                ) + R
            ) 
            # K = (B^T V B + R)^{-1} B^T V A 
            K[i] = torch.matmul(
                    torch.matmul(
                        torch.matmul(V_uu_inv,  B_trans),
                        V
                    ),
                    A
                )

            k[i] = utils.batch_mv(torch.matmul(V_uu_inv,  B.transpose(-2, -1)), v)

            #Ricatti difference equation
            # A-BK
            A_BK = A - torch.matmul(B, K[i])
            # V = A^T V (A-BK) + Q = A^T V A - A^T V B (B^T V B + R)^{-1} B^T V A + Q
            V = torch.matmul(torch.matmul(A_trans, V), A_BK) + Q
            # v = (A-BK)^Tv + Q r
            v = utils.batch_mv(A_BK.transpose(-2, -1), v) + utils.batch_mv(Q, goals[:, i, :])   

        # we might need to cat or stack to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k

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
        AB_cat = torch.bmm(utils.batch_pinv(GU_cat, I_factor, use_gpu=next(self.parameters()).is_cuda), G_next)

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
    
    def forward(self, x0):
        '''
        perform mpc with current parameters given the initial x0
        '''
        Q = torch.diag(self._q_diag_log.exp()).unsqueeze(0)
        R = torch.diag(self._r_diag_log.exp()).unsqueeze(0)
        if self._x_goal is not None:
            #this might have efficiency issue since the goal needs to be populated every call?
            goals = torch.repeat_interleave(self._phi(self._x_goal).unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
        else:
            #use g_goal instead
            assert(self._g_goal is not None)
            goals = torch.repeat_interleave(self._g_goal.unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
            
        K, k = self._solve_lqr(self._phi_affine.unsqueeze(0), self._u_affine.unsqueeze(0), Q, R, goals)
        #apply the first control as mpc
        # print(K[0].shape, k[0].shape)
        u = -utils.batch_mv(K[0], self._phi(x0)) + k[0] 
        return u