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


## Conditional VAE model (CVAE) -- p(x|z,y) with y observed.


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
        latent_dim = args.latent_dim # size of z space, without y label
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoder
        layers_dec = [(latent_dim+1, 50), (50, 100), (100, 200)]  # decoder

        # define type of variational encoder q(z|.)
        if args.use_q_z_y:
            # use q(z|x,y) -- standard for CVAE
            enc_args_z = {
                "layers_list": layers_enc,
                "activ": self.activ,
                "add_in_len": 1,  # for y tag
            }
        else:
            # use q(z|x) -- for testing purposes
            enc_args_z = {
                "layers_list": layers_enc,
                "activ": self.activ,
            }

        self.var_q_z = encodeMVNDiag(
            dataset_obj, args, self.latent_dim, mod_torso_enc=None, **enc_args_z
        )

        # p(z)
        _static_prior_z = {
            "mu": torch.zeros(self.latent_dim),
            "logvar": (
                (args.fixed_prior_zy1_sigma ** 2) * torch.ones(self.latent_dim)
            ).log(),
        } # "logvar": torch.ones(self.latent_dim).log(),
        self.prior_z = GaussDiagDistModule(self.latent_dim, static_params=_static_prior_z)

        # p(x|z)
        self.decoder = baseDecoder(dataset_obj, args, layers_dec, self.activ)

        # aux for storing info. later
        self.loss_ret_names = [
            "total_loss",
            "loss_elbo",
            "loss_sup", # cross-entropy; returns zeros since not used by CVAE.
            "nll",
            "kld_tot",
            "kld_z",
        ]

    def get_z_dists(self):

        """ Used for its r_sample and log_q functions alone """

        mdl_dists = dict()
        mdl_dists["q_z"] = self.var_q_z.param_nn
        mdl_dists["p_z"] = self.prior_z

        return mdl_dists

    def forward(self, x_data, n_epoch=None, y_targets=None, repair_mode=False):

        _device_used = x_data.device
        _len_batch = x_data.shape[0]

        q_params = dict()
        q_samples = dict()
        log_q_dists = dict()

        if y_targets is not None:
            _y_targets = y_targets.float().view(-1, 1)
        else:
            _y_targets = None

        if self.args.use_q_z_y:
            # use q(z|x,y) -- standard CVAE
            q_params["z"], q_samples["z"], _ = self.var_q_z(
                x_data,
                add_in=_y_targets,
                sampling=True,
                evalprob=False,
            )
        else:
            # use q(z|x) -- testing purpose
            q_params["z"], q_samples["z"], _ = self.var_q_z(
                x_data,
                sampling=True,
                evalprob=False,
            )

        q_samples["z_y1"] = torch.cat(
            [q_samples["z"], torch.ones((_len_batch, 1)).to(_device_used)], dim=1
        )
        q_samples["z_y0"] = torch.cat(
            [q_samples["z"], torch.zeros((_len_batch, 1)).to(_device_used)], dim=1
        )

        p_params = dict()
        p_params["z"], _, _ = self.prior_z()

        p_params["recon_y1"] = self.decoder(q_samples["z_y1"])
        p_params["recon_y0"] = self.decoder(q_samples["z_y0"])

        if (_y_targets is not None) and (not repair_mode):

            p_params["recon"] = dict()
            p_params["recon"]["x"] = (
                _y_targets * p_params["recon_y1"]["x"]
                + (1 - _y_targets) * p_params["recon_y0"]["x"]
            )

            if ("logvar_x" in p_params["recon_y1"]) and isinstance(
                p_params["recon_y1"]["logvar_x"], torch.Tensor
            ):
                # parameters for logvar is the same for y1 and y0, since uses same decoder p(x|z,y).
                p_params["recon"]["logvar_x"] = p_params["recon_y1"]["logvar_x"]

        else:
            # repair mode
            p_params["recon"] = p_params["recon_y1"]

        return (p_params, q_params, q_samples, log_q_dists)

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