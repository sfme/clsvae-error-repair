import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torch.distributions as dist
from collections import OrderedDict

from repair_syserr_models.module_utils import (
    baseEncoder,
    baseDecoder,
    GaussDiagDistModule,
)

from repair_syserr_models.model_utils import logit_fn
from repair_syserr_models.loss_utils import nll_batch_noreduce


EPS = 1e-8

#########
#    Semi-supervised CCVAE model; SOTA baseline.
#    
#    Original code from: Tom Joy
#       paper: https://arxiv.org/pdf/2006.10102.pdf
#       code: https://github.com/thwjoy/ccvae
#       
#    Modified code by: Simao Eduardo
#########


class Diagonal(nn.Module):
    def __init__(self, dim):

        super(Diagonal, self).__init__()

        self.dim = dim
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):

        return x * self.weight.view(1, -1) + self.bias.view(1, -1)


class Classifier(nn.Module):
    def __init__(self, dim):

        super(Classifier, self).__init__()

        self.dim = dim
        self.diag = Diagonal(self.dim)

    def forward(self, x):

        return self.diag(x)


class CondPrior(nn.Module):
    def __init__(self, dim):

        super(CondPrior, self).__init__()

        self.dim = dim
        self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
        self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
        self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
        self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

    def forward(self, x):

        _diag_loc_true = self.diag_loc_true.view(1, -1)
        _diag_loc_false = self.diag_loc_false.view(1, -1)

        _diag_scale_true = self.diag_scale_true.view(1, -1)
        _diag_scale_false = self.diag_scale_false.view(1, -1)

        loc = x * _diag_loc_true + (1 - x) * _diag_loc_false
        scale = x * _diag_scale_true + (1 - x) * _diag_scale_false

        return loc, torch.clamp(F.softplus(scale), min=1e-3)


class encMod(nn.Module):
    def __init__(self, dataset_obj, args, layers_enc, latent_dim, activ):

        super().__init__()

        self.z_dim = latent_dim
        self.encoder = baseEncoder(dataset_obj, args, layers_enc, activ)

        self.locs = nn.Linear(self.encoder.layers_list[-1][1], self.z_dim)
        self.scales = nn.Linear(self.encoder.layers_list[-1][1], self.z_dim)

    def forward(self, x):

        h_last = self.encoder(x)
        return self.locs(h_last), torch.clamp(F.softplus(self.scales(h_last)), min=1e-3)


def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)


class VAE(nn.Module):
    """
    CCVAE -- Tom Joy et. al.
    """

    def __init__(self, dataset_obj, args):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        self.args.size_input = len(
            dataset_obj.cat_cols
        ) * self.args.embedding_size + len(dataset_obj.num_cols)

        self.args.size_output = len(dataset_obj.cat_cols) + len(
            dataset_obj.num_cols
        )  # 2*

        if args.activation == "relu":
            self.activ = nn.ReLU()
        elif args.activation == "hardtanh":
            self.activ = nn.Hardtanh()
        elif args.activation == "selu":
            self.activ = nn.SELU()

        # vae neural network architecture (encoders and decoder)
        self.latent_dim = args.latent_dim + 1
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoders
        layers_dec = [(self.latent_dim, 50), (50, 100), (100, 200)]  # decoder

        self.encoder = encMod(dataset_obj, args, layers_enc, self.latent_dim, self.activ)
        self.decoder = baseDecoder(dataset_obj, args, layers_dec, self.activ)

        ##

        self.z_dim = self.latent_dim
        self.z_classify = 1  # was: num_classes
        self.z_style = self.latent_dim - self.z_classify

        self.use_cuda = args.cuda_on
        self.num_classes = 1  # was: num_classes

        self.register_buffer("ones", torch.ones(1, self.z_style))
        self.register_buffer("zeros", torch.zeros(1, self.z_style))

        self.register_buffer(
            "y_prior_params", torch.tensor(args.y_clean_prior).view(1, 1)
        )

        ##

        self.classifier = Classifier(self.num_classes)
        self.cond_prior = CondPrior(self.num_classes)

        self._buffer_loss_unsup = []
        self._buffer_nll_unsup = []
        self._buffer_kl_unsup = []

        self._buffer_loss_sup = []
        self._buffer_nll_sup = []
        self._buffer_kl_sup = []

        # aux for storing info. later
        self.loss_ret_names = [
            "total_loss",
            "loss_elbo",
            "loss_sup",
            "nll",
            "kld_tot",
            "kld_z",
        ]

    def get_z_dists(self):

        """ Used for its r_sample and log_q functions alone """

        mdl_dists = dict()
        mdl_dists["q_z"] = GaussDiagDistModule(self.z_dim)
        mdl_dists["p_z"] = GaussDiagDistModule(self.z_dim)

        # NOTE: q_z same type of Gaussian dist for both y labels, since encoder NN will have info on y label.
        mdl_dists["q_z_y0"] = GaussDiagDistModule(self.z_dim)
        mdl_dists["q_z_y1"] = GaussDiagDistModule(self.z_dim)

        mdl_dists["p_z_y0"] = GaussDiagDistModule(self.z_dim)
        mdl_dists["p_z_y1"] = GaussDiagDistModule(self.z_dim)

        return mdl_dists

    def log_likelihood(self, recon_params, input_data):
        return -1 * nll_batch_noreduce(
            self.dataset_obj, input_data, recon_params
        )  # Bx1

    def unsup(self, x):

        bs = x.shape[0]

        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Bernoulli(logits=self.classifier(zc))
        y = qyzc.sample()
        log_qy = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (
            torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1),
            torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1),
        )
        kl = compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon_params = self.decoder(z)
        log_py = (
            dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y).sum(dim=-1)
        )

        log_pxz = self.log_likelihood(recon_params, x).sum(dim=-1)

        elbo = log_pxz + log_py - kl - log_qy  # .mean()

        return -elbo, -log_pxz, kl

    def sup(self, x, y):

        _upsamp_coeff = self.args.q_y_x_coeff

        bs = x.shape[0]

        # inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Bernoulli(logits=self.classifier(zc))
        log_qyzc = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (
            torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1),
            torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1),
        )
        kl = compute_kl(*post_params, *prior_params)

        # compute log probs for x and y
        recon_params = self.decoder(z)

        log_py = (
            dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y).sum(dim=-1)
        )

        log_qyx = self.classifier_loss(x, y)

        log_pxz = self.log_likelihood(recon_params, x).sum(dim=-1)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_ = (
            dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(y).sum(dim=-1)
        )
        w = torch.exp(log_qyzc_ - log_qyx)
        elbo = (
            w * (log_pxz - kl - log_qyzc) + log_py + _upsamp_coeff * log_qyx
        )  # .mean()

        return -elbo, -log_pxz, kl

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """

        zc, _ = (
            dist.Normal(*self.encoder(x))
            .rsample(torch.tensor([k]))
            .split([self.z_classify, self.z_style], -1)
        )
        logits = self.classifier(zc.view(-1, self.z_classify))
        d = dist.Bernoulli(logits=logits)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

    def eval_repair_fwd(self, x):
        """
        Function used for evaluation on train / test data; and for data repair process.

        # p_params --> recon params (softmax probs and gaussian per feat.), and prior information
        # q_params --> mu, logvar of gauss variational dist
        # q_samples --> samples from gauss variational dist

        """

        with torch.no_grad():

            bs = x.shape[0]

            _device_used = x.device

            ## encode and params
            post_params = self.encoder(x)

            _mu_zc, _mu_zs = post_params[0].split([self.z_classify, self.z_style], 1)
            _logvar_zc, _logvar_zs = post_params[1].split(
                [self.z_classify, self.z_style], 1
            )
            _logvar_zc, _logvar_zs = _logvar_zc.log() * 2.0, _logvar_zs.log() * 2.0

            _ones_vec = torch.ones(bs, 1).to(_device_used)
            _mu_pzc_y1, _logvar_pzc_y1 = self.cond_prior(_ones_vec)  # y
            _logvar_pzc_y1 = _logvar_pzc_y1.log() * 2.0

            _zeros_vec = torch.zeros(bs, 1).to(_device_used)
            _mu_pzc_y0, _logvar_pzc_y0 = self.cond_prior(_zeros_vec)  # y
            _logvar_pzc_y0 = _logvar_pzc_y0.log() * 2.0

            _logits_qyzc = self.classifier(_mu_zc)

            ## samples
            z = dist.Normal(*post_params).rsample()

            # get eval / repair structs
            p_params = dict()
            q_params = dict()
            q_samples = dict()
            log_q_dists = dict()

            q_params["y"] = dict()
            q_params["y"]["logits"] = _logits_qyzc
            
            # NOTE: populate with the same z representation and recons for y=1 and y=0, though not used in practice later.
            q_params["z_y1"] = dict()
            q_params["z_y1"]["mu"] = post_params[0]
            q_params["z_y1"]["logvar"] = post_params[1].log() * 2.0

            q_params["z_y0"] = dict()
            q_params["z_y0"]["mu"] = post_params[0]
            q_params["z_y0"]["logvar"] = post_params[1].log() * 2.0

            p_params["recon_y1"] = self.decoder(post_params[0])
            p_params["recon_y0"] = p_params["recon_y1"]

            q_samples["z_y1"] = z
            q_samples["z_y0"] = z

            # NOTE: this will be used in for either evaluation, or the data repair process.
            q_params["z"] = dict()
            q_params["z"]["mu"] = torch.cat([_mu_pzc_y1, _mu_zs], 1)  # z params for y=1
            q_params["z"]["logvar"] = torch.cat([_logvar_pzc_y1, _logvar_zs], 1)  # z params for y=1

            p_params["recon"] = self.decoder(q_params["z"]["mu"]) # recon for y=1 (proposed repair)

            # return prior parameters for y and z | y 
            p_params["y"] = dict()
            p_params["y"]["logits"] = logit_fn(self.y_prior_params.flatten())

            p_params["z_y1"] = dict()
            p_params["z_y1"]["mu"] = torch.cat(
                [self.cond_prior.diag_loc_true, self.zeros.flatten()]
            )
            p_params["z_y1"]["logvar"] = torch.cat(
                [self.cond_prior.diag_scale_true.log() * 2, self.zeros.flatten()]
            )

            p_params["z_y0"] = dict()
            p_params["z_y0"]["mu"] = torch.cat(
                [self.cond_prior.diag_loc_false, self.zeros.flatten()]
            )
            p_params["z_y0"]["logvar"] = torch.cat(
                [self.cond_prior.diag_scale_false.log() * 2, self.zeros.flatten()]
            )

            return p_params, q_params, q_samples, log_q_dists

    def forward(self, x_data, n_epoch=None, y_targets=None, repair_mode=False):

        """
        -> The losses used in .backward() are stored in the buffers; the function return is for evaluation purposes.

        -> Two modes: i) for evaluation mode for train / test datasets; ii) for data repair process (no labels used).
        """

        if (y_targets is not None) and (not repair_mode):
            # needed in training mode, where losses stored in buffers.

            _y_targets = y_targets.float().view(-1, 1)

            ret_unsup = self.unsup(x_data)
            self._buffer_loss_unsup = ret_unsup[0]
            self._buffer_nll_unsup = ret_unsup[1]
            self._buffer_kl_unsup = ret_unsup[2]

            ret_sup = self.sup(x_data, _y_targets)
            self._buffer_loss_sup = ret_sup[0]
            self._buffer_nll_sup = ret_sup[1]
            self._buffer_kl_sup = ret_sup[2]

            return self.eval_repair_fwd(x_data)

        else:
            # used for model repair eval only

            # clear buffers
            self._buffer_loss_unsup = []
            self._buffer_nll_unsup = []
            self._buffer_kl_unsup = []

            self._buffer_loss_sup = []
            self._buffer_nll_sup = []
            self._buffer_kl_sup = []

            return self.eval_repair_fwd(x_data)

    def loss_function(
        self,
        input_data,
        p_params,
        q_params,
        q_samples,
        log_q_dists,
        mask_semi=None,
        y_targets=None,
        sup_coeff=1.0,
        kl_coeff=1.0,
        reg_scheduler_val=None,
        n_epoch=None,
    ):

        """
        NOTE: actual losses for backwards are stored in the buffers; and then these buffers should be cleansed after use.

        """

        _device_used = input_data.device
        _batch_len = input_data.shape[0]

        if (mask_semi is not None) and (y_targets is not None):
            _mask_semi = mask_semi.view(-1, 1).float()

        else:
            _mask_semi = torch.zeros(_batch_len, 1).to(_device_used)

        _mask_semi = _mask_semi.flatten()

        total_loss = (
            self._buffer_loss_sup * _mask_semi
            + self._buffer_loss_unsup * (1.0 - _mask_semi)
        ).sum()

        nll = (
            self._buffer_nll_sup * _mask_semi
            + self._buffer_nll_unsup * (1.0 - _mask_semi)
        ).sum()

        kld_z = (
            self._buffer_kl_sup * _mask_semi
            + self._buffer_kl_unsup * (1.0 - _mask_semi)
        ).sum()

        # clear buffers
        self._buffer_loss_unsup = []
        self._buffer_nll_unsup = []
        self._buffer_kl_unsup = []

        self._buffer_loss_sup = []
        self._buffer_nll_sup = []
        self._buffer_kl_sup = []

        ret_dict = OrderedDict(
            [
                ("total_loss", total_loss),
                ("loss_elbo", total_loss),
                ("loss_sup", torch.tensor(0.0, device=_device_used)),
                ("nll", nll),
                ("kld_tot", kld_z),
                ("kld_z", kld_z),
            ]
        )

        return ret_dict
