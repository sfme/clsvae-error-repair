import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dists
import numpy as np

EPS = 1e-8


def logit_fn(x):
    return (x + EPS).log() - (1.0 - x + EPS).log()


def trf_param_to_cholesky(latent_dim, param_tensor):

    """
    inputs: param_tensor (BxN or 1xN) is batch tensor where 2nd dim holds cholesky matrix
    entries. The first 'latent_dim' elements in 2nd dim are diagonal entries
    (i.e. >0), the rest are lower triangular matrix elements (below diagonal).

    NOTE: number of non-zero cholesky entries is n*(n+1)/2

    outputs: 1xDxD or BxDxD
    """

    _device = param_tensor.device
    _shape = param_tensor.shape

    n_elems_chol = latent_dim * (latent_dim + 1) / 2
    assert _shape[-1] == n_elems_chol

    if len(_shape) < 2:
        _param_tensor = param_tensor.view(1, -1)
    else:
        _param_tensor = param_tensor

    # diagonal -- init. chol matrix, already with diagonal elements
    chol_lower = torch.diag_embed(F.softplus(_param_tensor[:, :latent_dim]) + EPS)

    # non-diagonal elements
    idxs_lower = torch.tril_indices(latent_dim, latent_dim, offset=-1, device=_device)
    chol_lower[:, idxs_lower[0], idxs_lower[1]] = _param_tensor[:, latent_dim:]

    return chol_lower


def log_mean_exp(inputs, dim=1):

    input_max = inputs.max(dim, keepdim=True)[0]
    return (inputs - input_max).exp().mean(dim=dim, keepdim=True).log() + input_max


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""

    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""

    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps, collapse=True):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""

    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    if collapse:
        return merge_leading_dims(x, num_dims=2)
    else:
        return x


class DiagScaleMatrix(nn.Module):
    def __init__(self, var_dim):
        super().__init__()

        self.var_dim = var_dim

        self.A_vec = nn.Parameter(torch.randn(var_dim))

    def apply_fwd(self, in_var):
        return self.A_vec.view(1, -1) * in_var

    def apply_inv(self, s_var):
        return (1 / self.A_vec).view(1, -1) * s_var

    def forward(self, in_var):
        return self.apply_fwd(in_var)


## Dropout and Masking (e.g. can use in encoder network)

def masker(logit_pi, gs_temp=1, stochastic=True, dropout=False, out_shape=[]):

    """
    inputs:
        -- logit_pi : BxD (D is feature lengths, can be 1)

    outputs:
        -- drop_mask: (N,D) defines which entries are to be zeroed-out
    """

    if dropout:
        # note that here logit_pi only has one value for the entire batch / features
        p_dropout = 1.0 - torch.sigmoid(logit_pi)

        if stochastic:
            return (
                torch.empty(out_shape, device=logit_pi.device).uniform_() > p_dropout
            ).float()
        else:
            return torch.ones(out_shape, device=logit_pi.device).float()

    else:

        if stochastic:
            gumbel_dist_noise = dists.gumbel.Gumbel(
                torch.tensor([0.0]), torch.tensor([1.0])
            )
            g_noise = (
                gumbel_dist_noise.sample(sample_shape=logit_pi.shape)
                .type(logit_pi.type())
                .squeeze(dim=-1)
            )
        else:
            g_noise = 0.0

        inner = (logit_pi + g_noise) / gs_temp
        samples = torch.sigmoid(inner)

        samples_hard = torch.round(samples)

        return (samples_hard - samples).detach() + samples



## Schedulers for KL annealing

def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch) * stop
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1

    return L
