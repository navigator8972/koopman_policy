"""
Solving LQR for Koopman embedded dynamical systems. Keep everything differentiable.
"""

import torch
from torch import nn, optim

import utils

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class KoopmanLQR(nn.Module):
    def __init__(self, k, x_dim, u_dim, x_goal, T, phi=None, u_affine=None):
        """
        k:          rank of approximated koopman operator
        x_dim:      dimension of system state
        u_dim:      dimension of control input
        x_goal:     goal state
        T:          length of horizon
        phi:        observation func for state. nn.Module. should be a module handling input dim x_dim and output k dim vectors
                    use a linear subspace projection if kept None
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        """
        super().__init__()
        self._k = k
        self._x_dim = x_dim
        self._u_dim = u_dim
        self._x_goal = x_goal
        self._T = T

        # Koopman observable function: putting state and control together as an augmented autonomous state
        # and assume control is not subject a dynamical process h(x_t+1, 0) = K h(x_t, u_t)
        # only state part is nonlinearly embedded: h(x_t, u_t) = phi(x_t) + Cu_t
        # put together: h(x_t+1, 0) = A phi(x_t) + Bu_t --> phi(x_t+1) = A phi(x_t) + B u_t, this makes a linear system w.r.t. g(.)
        if phi is None:
            self._phi = nn.Linear(x_dim, k)
        elif phi == 'FCNN':
            #use a simple fully connected neural network
            self._phi = FCNN(x_dim, k, 16)
        else:
            self._phi = phi
        
        #prepare linear system params
        self._phi_affine = nn.Linear(k, k)

        if u_affine is None:
            self._u_affine = nn.Linear(u_dim, k)
        else:
            self._u_affine = u_affine
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
            V_uu_inv = torch.pinverse(
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

    def fit_koopman(self, X, U, train_phi=True, n_itrs=100, lr=1e-4, verbose=False):
        '''
        fit koopman paramters, phi, A and B
        X, U:   trajectories of states and actions (batch_size, T, dim), self-supervision by ignoring the last action
        n_itrs: number of optimization iterations
        '''

        u_curr = U[:, :X.shape[1]-1, :]

        #choose to train NN feature or not. In the case of False, the embedding is basically a random projection
        if train_phi:
            params = list(self._phi.parameters()) + list(self._phi_affine.parameters()) + list(self._u_affine.parameters())
        else:
            #this can actually solved by pinverse...
            params = list(self._phi_affine.parameters()) + list(self._u_affine.parameters())
        
        opt = optim.Adam(params, lr=lr, weight_decay=0)
        loss = nn.MSELoss()
        
        for i in range(n_itrs):
            g_next = self._phi(X[:, 1:, :])
            g_curr = self._phi(X[:, :-1, :])
            
            g_pred = self._phi_affine(g_curr)+self._u_affine(u_curr)
            #eval loss over all of them

            output = loss(g_pred, g_next)

            #apply gradient
            opt.zero_grad()
            output.backward()
            opt.step()

            if verbose:
                print('Iteration {0}: MSE loss - {1}'.format(i, output.item()))
        return
    
    def predict_koopman(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        return self._phi_affine(G)+self._u_affine(U)