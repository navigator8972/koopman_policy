
import torch
from torch import nn

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

    x_t = torch.transpose(x, 1, 2)

    # use_gpu = torch.cuda.is_available()
    # use_gpu = False #test for now...
    I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
    if use_gpu:
        I = I.cuda()

    x_pinv = torch.bmm(
        torch.inverse(torch.bmm(x_t, x) + I_factor * I),
        x_t
    )

    if trans:
        x_pinv = torch.transpose(x_pinv, 1, 2)

    return x_pinv



# net = OrthogonalOutputFCNN(4, 32, 4, [32, 32])
# print(net.orthogonal_transpose.weight.T @ net.orthogonal_transpose.weight)
# print(net)
# test_input = torch.randn((128, 4))
# print(net(test_input))