
import torch

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