import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from repair_syserr_models.loss_utils import semi_y_vae_ELBO
from repair_syserr_models.module_utils import (
    baseEncoder,
    baseDecoder,
    encodeMVNDiag,
    GaussDiagDistModule,
    BernoulliDistModule,
    encodeBern,
)

EPS = 1e-8

## Semi-Supervised VAE, with 2-component GMM in z-space -- p(x|z) p(z|y) p(y)


class VAE(nn.Module):
    def __init__(self, dataset_obj, args):

        super().__init__()
        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)

        self.dataset_obj = dataset_obj
        self.args = args

        # boolean: should the 2-component GMM prior be learnt? (True) or should be static? (False)
        self.learned_prior_comps = args.learn_z_given_y_priors

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
        self.latent_dim = args.latent_dim
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoders
        layers_dec = [(self.latent_dim, 50), (50, 100), (100, 200)]  # decoder

        # q(y|x)
        enc_args_y = {"layers_list": layers_enc, "activ": self.activ}
        self.var_q_y = encodeBern(
            dataset_obj, args, mod_torso_enc=None, **enc_args_y
        )

        # q(z|y,x)
        enc_args_z = {
            "layers_list": layers_enc,
            "activ": self.activ,
            "add_in_len": 1,  # for y encoding!
        }
        self.var_q_z = encodeMVNDiag(
            dataset_obj, args, self.latent_dim, mod_torso_enc=None, **enc_args_z
        )

        # p(y) -- prior on ratio of inliers in the data
        _static_prior_y = torch.tensor(args.y_clean_prior)
        self.prior_y = BernoulliDistModule(static_prob=_static_prior_y)

        # p(z|y) learned prior distribution
        if self.learned_prior_comps:
            # p(z|y=1)
            self.prior_z_y1 = GaussDiagDistModule(self.latent_dim)

            # p(z|y=0)
            self.prior_z_y0 = GaussDiagDistModule(self.latent_dim)

        else:
            # p(z|y=1) static prior distribution
            _static_prior_z_y1 = {
                "mu": torch.zeros(self.latent_dim),
                "logvar": (
                    (args.fixed_prior_zy1_sigma ** 2) * torch.ones(self.latent_dim)
                ).log(),
            }
            self.prior_z_y1 = GaussDiagDistModule(
                self.latent_dim, static_params=_static_prior_z_y1
            )

            # p(z|y=0)
            _static_prior_z_y0 = {
                "mu": torch.zeros(self.latent_dim),
                "logvar": (
                    (args.fixed_prior_zy0_sigma ** 2) * torch.ones(self.latent_dim)
                ).log(),
            }
            self.prior_z_y0 = GaussDiagDistModule(
                self.latent_dim, static_params=_static_prior_z_y0
            )

        # p(x|z)
        self.decoder = baseDecoder(dataset_obj, args, layers_dec, self.activ)

        # aux for storing info. later
        self.loss_ret_names = [
            "total_loss",
            "loss_elbo",  # data ELBO
            "loss_sup",  # classifier loss.
            "nll",
            "kld_tot",
            "kld_y",
            "kld_z_y1",
            "kld_z_y0",
        ]

    def get_z_dists(self):

        """ Used for its r_sample and log_q functions alone """

        mdl_dists = dict()
        mdl_dists["q_z"] = self.var_q_z.param_nn
        mdl_dists["p_z"] = self.prior_z_y1

        # NOTE: q_z same type of Gaussian Dist both y's, since func_nn will have info on y label.
        mdl_dists["q_z_y0"] = self.var_q_z.param_nn
        mdl_dists["q_z_y1"] = self.var_q_z.param_nn

        mdl_dists["p_z_y0"] = self.prior_z_y0
        mdl_dists["p_z_y1"] = self.prior_z_y1

        return mdl_dists

    def forward(self, x_data, n_epoch=None, y_targets=None, repair_mode=False):

        # p_params --> recon params (softmax probs and gaussian per feat.), and prior params
        # q_params --> mu, logvar of gauss variational dist
        # q_samples --> samples from gauss variational dist

        _device_used = x_data.device

        q_params = dict()
        q_samples = dict()
        log_q_dists = dict()

        # forward q(y | x)
        q_params["y"], _, _ = self.var_q_y(x_data, sampling=False, evalprob=False)

        # NOTE: to be used in: supervised part of loss; and outlier detection (OD) analytics.
        if y_targets is None:
            log_q_dists["y"] = []
        else:
            _y_targets = y_targets.float()
            log_q_dists["y"] = self.var_q_y.param_nn.log_probs(
                _y_targets, force_params=q_params["y"]
            )

        # forward q(z | x, y=1)
        pick_y = torch.ones(x_data.shape[0], 1).to(_device_used)
        q_params["z_y1"], q_samples["z_y1"], _ = self.var_q_z(
            x_data, add_in=pick_y, sampling=True, evalprob=False
        )

        # forward q(z | x, y=0)
        pick_y = torch.zeros(x_data.shape[0], 1).to(_device_used)
        q_params["z_y0"], q_samples["z_y0"], _ = self.var_q_z(
            x_data, add_in=pick_y, sampling=True, evalprob=False
        )

        p_params = dict()
        p_params["y"], _, _ = self.prior_y()

        p_params["z_y1"], _, _ = self.prior_z_y1()
        p_params["z_y0"], _, _ = self.prior_z_y0()

        p_params["recon_y1"] = self.decoder(q_samples["z_y1"])
        p_params["recon_y0"] = self.decoder(q_samples["z_y0"])

        # NOTE: to be used in eval / repair; uses MAP solution from clean component.
        q_params["z"] = q_params["z_y1"]
        q_samples["z"] = q_samples["z_y1"]
        p_params["recon"] = p_params["recon_y1"]
        p_params["z"] = p_params["z_y1"]

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

        _device_used = input_data.device
        _batch_len = input_data.shape[0]

        if (mask_semi is not None) and (y_targets is not None):
            _mask_semi = mask_semi.view(-1, 1)
            _y_targets = y_targets.view(-1, 1)
        else:
            _mask_semi = None
            _y_targets = None

        ret_list = semi_y_vae_ELBO(
            self.dataset_obj,
            input_data,
            p_params,
            q_params,
            _y_targets,
            _mask_semi,
            kl_coeff=kl_coeff,
            q_samples=q_samples,
        )

        (
            elbo_loss,
            nll,
            kld_tot,
            kld_y,
            kld_z_y1,
            kld_z_y0,
        ) = ret_list

        # supervision loss (e.g. classifier)
        if y_targets is None or mask_semi is None:
            _sup_len = torch.tensor(1.0).to(_device_used)
            sup_loss = torch.tensor(0.0).to(_device_used)
        else:
            _sup_len = _mask_semi.sum()
            # cross-entropy for classifier
            sup_loss = -(log_q_dists["y"] * _mask_semi).sum()

        if _sup_len > 0:
            sup_loss_scaled = (_batch_len / float(_sup_len)) * sup_loss
        else:
            sup_loss = torch.tensor(0.0).to(_device_used)
            sup_loss_scaled = torch.tensor(0.0).to(_device_used)

        total_loss = elbo_loss + sup_coeff * sup_loss_scaled

        ret_dict = OrderedDict(
            [
                ("total_loss", total_loss),
                ("loss_elbo", elbo_loss),
                ("loss_sup", sup_loss),
                ("nll", nll),
                ("kld_tot", kld_tot),
                ("kld_y", kld_y),
                ("kld_z_y1", kld_z_y1),
                ("kld_z_y0", kld_z_y0),
            ]
        )

        return ret_dict
