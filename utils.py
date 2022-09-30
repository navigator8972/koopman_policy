
from turtle import forward
import torch
from torch import nn

class KoopmanThinPlateSplineBasis(nn.Module):
    """
    thin plate spline radial basis function
    as used in arxiv/1611.03537 
    """
    def __init__(self, in_dim, n_basis, center_dist_box_scale=1.) -> None:
        super().__init__()
        #prepare n_basis function with centers sampled from the box of a scale of center_dist_box_scale
        #the output dimension is of a length in_dim+n_basis with original features augmented
        self.register_buffer('_centers', (torch.rand(n_basis, in_dim)*2 - 1) * center_dist_box_scale)

        self.in_dim = in_dim
        self.n_basis = n_basis

    def forward(self, x):
        """
        x       - leading batch dimensions, ..., in_dim

        z       - leading batch dimensions, ..., in_dim+n_basis
        """
        # print(x.unsqueeze(-2).shape, self._centers.unsqueeze(0).shape)
        radial_square = torch.sum(torch.sub(x.unsqueeze(-2), self._centers)**2, dim=-1)
        phi = 0.5 * radial_square * torch.log(radial_square)

        return torch.cat((x, phi), dim=-1)
    
    def inverse(self, z):
        #return the first in_dim dimension of z
        return z[:, :self.in_dim]
    


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
            modules_dim = [in_dim]+hidden_dim
            for idx in range(len(modules_dim)-1):
                modules.append(nn.Linear(modules_dim[idx], modules_dim[idx+1]))
                if hidden_nonlinearity is None:
                    modules.append(nn.ReLU())
                else:
                    modules.append(hidden_nonlinearity())
            modules.append(nn.Linear(modules_dim[-1], out_dim))
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

class KoopmanFCNNLift(nn.Module):
    """
    An FCNN lift function with skip output allowing to recover the original input
    input:  x
    output: [x; FCNN(x)] - so inverse could be just first input size dimensions
    """
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_nonlinearity=None, output_nonlinearity=None):
        super().__init__()
        #note here out_dim should be larger than in_dim so we can automatically determine the NN output layer
        assert(in_dim < out_dim)
        self.fcnn_ = FCNN(in_dim, out_dim-in_dim, hidden_dim, hidden_nonlinearity, output_nonlinearity)
        self.in_dim_ = in_dim
        self.out_dim_ = out_dim
    
        self.fcnn_.init_params()
    
    def forward(self, x):
        return torch.cat((x, self.fcnn_(x)), dim=-1)
    
    def inverse(self, z):
        return z[..., :self.in_dim_]

# comment out for now because the torch version conflict between garage and geotorch (1.51 vs 1.80)
# import geotorch
# import torch.nn.functional as F 

# class OrthogonalOutputFCNN(FCNN):
#     """
#     FCNN with output layer, this stacks an orthogonal linear layer on top of an FCNN
#     so the net would be (in_dim, hidden_dim, out_dim, orthoutput_dim)
#     """
#     def __init__(self, in_dim, out_dim, orthoutput_dim, hidden_dim, hidden_nonlinearity=None):
#         #no output nonlinearity as we expect an orthogonal linear transformation
#         super().__init__(in_dim, out_dim, hidden_dim, hidden_nonlinearity, None)
#         #use transpose of mat instead of linear because it is more convenient for geotorch to enforce column orthogonality
#         self.orthogonal_transpose = nn.Linear(orthoutput_dim, out_dim, bias=False)
#         geotorch.constraints.orthogonal(self.orthogonal_transpose, 'weight')
#         return
    
#     #not sure if we should have a separate initialization for the orthogonal output layer
#     def init_params(self):
#         super().init_params()
#         # self.orthogonal_transpose.sample()
#         return
    
#     def forward(self, x):
#         with geotorch.parametrize.cached():
#             fcnn_output = self.network(x)
#             #apply this to orthogonal transpose as well
#             out = F.linear(fcnn_output, self.orthogonal_transpose.weight.T)
#         return out

def batch_mv(bmat, bvec):
    r"""
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing :math:`n \times n` matrices, and
    `bvec`, containing length :math:`n` vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)

#from compositional koopman work
#curiously the default I_factor is 10, meaning we cannot really pursue an accurate least square fit because backpropagate of inv is unstable?
def batch_pinv(x, I_factor, use_gpu=False):

    """
    :param x: B x N x D (N > D)
    :param I_factor:
    :return:
    """

    B, N, D = x.size()

    if N < D:
        x = torch.transpose(x, 1, 2)
        N, D = D, N
        trans = True
    else:
        trans = False
    
    I = torch.eye(D)[None, :, :].repeat(B, 1, 1)

    x_t = torch.transpose(x, 1, 2)

    if use_gpu:
        I = I.cuda()


    #for x_pinv@x == I
    #warning inverse is sensitive to float32/64, try to use pinv/lstsq as now these functions support batch version as well
    x_pinv = torch.bmm(
        torch.inverse(torch.bmm(x_t, x) + I_factor * I),
        x_t
    )

    if trans:
        x_pinv = torch.transpose(x_pinv, 1, 2)

    return x_pinv

def tanh_inv(x, epsilon=1e-6):
    pre_tanh_value = torch.log(
    (1 + epsilon + x) / (1 + epsilon - x)) / 2
    return pre_tanh_value

# net = OrthogonalOutputFCNN(4, 32, 4, [32, 32])
# print(net.orthogonal_transpose.weight.T @ net.orthogonal_transpose.weight)
# print(net)
# test_input = torch.randn((128, 4))
# print(net(test_input))

# soft update model from garage implementation
# pylint: disable=missing-param-doc, missing-type-doc
def soft_update_model(target_model, source_model, tau):
    """Update model parameter of target and source model.
    # noqa: D417
    Args:
        target_model
                torch.nn.Module
        source_model
                torch.nn.Module
                    Source network to update.
        tau (float): Interpolation parameter for doing the
            soft target update.
    """
    for target_param, param in zip(target_model.parameters(),
                                   source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)