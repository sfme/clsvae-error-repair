import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from repair_syserr_models.loss_utils import vae_standard_ELBO
from repair_syserr_models.module_utils import (
    baseEncoder,
    baseDecoder,
    encodeMVNDiag,
    GaussDiagDistModule,
)


EPS = 1e-8

## Standard VAE model -- p(x|z) p(z)


class VAE(nn.Module):
    def __init__(self, dataset_obj, args):

        super().__init__()

        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)

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

        # vae neural network architecture (encoder and decoder)
        self.latent_dim = args.latent_dim
        latent_dim = args.latent_dim
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoder
        layers_dec = [(latent_dim, 50), (50, 100), (100, 200)]  # decoder

        enc_args = {"layers_list": layers_enc, "activ": self.activ}
        self.var_q_z = encodeMVNDiag(
            dataset_obj, args, latent_dim, mod_torso_enc=None, **enc_args
        )

        static_prior = {
            "mu": torch.zeros(latent_dim),
            "logvar": torch.log(torch.ones(latent_dim)),
        }
        self.prior_z = GaussDiagDistModule(
            latent_dim, static_params=static_prior
        )  # will use register_buffer()

        self.decoder = baseDecoder(dataset_obj, args, layers_dec, self.activ)

        # aux for storing info. later
        self.loss_ret_names = [
            "total_loss",
            "loss_elbo",
            "loss_sup",
            "nll",
            "kld_tot",
            "kld_z",
        ]  # total_loss is elbo; and loss_sup is 0

    def get_z_dists(self):
        
        """ return dict used for its r_sample and log_q functions alone """

        mdl_dists = dict()
        mdl_dists["q_z"] = self.var_q_z.param_nn
        mdl_dists["p_z"] = self.prior_z

        return mdl_dists

    def forward(self, x_data, n_epoch=None, y_targets=None, repair_mode=False):

        # p_params --> recon params (softmax probs and gaussian per feat.), and prior params
        # q_params --> mu, logvar of gauss variational dist
        # q_samples --> samples from gauss variational dist

        q_params = dict()
        q_samples = dict()
        log_q = dict()

        # variational params, samples, and evals
        q_params["z"], q_samples["z"], log_q["z"] = self.var_q_z(
            x_data, sampling=True, evalprob=False
        )

        p_params = dict()  # decoder / prior params
        p_params["z"], _, _ = self.prior_z(sampling=False, evalprob=False)
        p_params["recon"] = self.decoder(q_samples["z"])

        return p_params, q_params, q_samples, []

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

        _device = input_data.device

        elbo_loss, nll, kld_z = vae_standard_ELBO(
            self.dataset_obj,
            input_data,
            p_params,
            q_params,
            kl_coeff=kl_coeff,
        )

        ret_dict = OrderedDict(
            [
                ("total_loss", elbo_loss),
                ("loss_elbo", elbo_loss),
                ("loss_sup", torch.tensor(0.0, device=_device)),
                ("nll", nll),
                ("kld_tot", kld_z),
                ("kld_z", kld_z),
            ]
        )

        return ret_dict
