import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from repair_syserr_models.loss_utils import semi_y_vae_partitioned_ELBO

from repair_syserr_models.module_utils import (
    baseEncoder,
    baseDecoder,
    GaussDiagDistModule,
    BernoulliDistModule,
    modSeq,
)

EPS = 1e-6

############################################
#
# CLSVAE model -- proposed by the author in paper https://arxiv.org/pdf/2207.08050.pdf
#
# generative model: p(x| z_c, z_d, z_\epsilon, y) p(z_c) p(z_d) p(z_\epsilon) p(y)
# 
############################################

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

        # vae neural network architecture (encoders and decoder)
        self.latent_dim = args.latent_dim # 15 is default
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoders
        layers_dec = [(self.latent_dim, 50), (50, 100), (100, 200)]  # decoder

        # NOTE: Use 1/3 of latent space for dirty representation (i.e. size of z_d), other options valid.
        self.z_dirty_dim = self.latent_dim // 3
        self.z_clean_dim = self.latent_dim - self.z_dirty_dim

        # q(y|z)
        # -- NN func
        self.q_y_logits_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            self.activ,
            nn.Linear(self.latent_dim // 2, self.latent_dim // 3),
            self.activ,
            nn.Linear(self.latent_dim // 3, 1),
        )
        # -- distribution
        self.q_y_dist = BernoulliDistModule(func_nn={"logits": self.q_y_logits_net})

        # q(z|x)
        _enc_z_layers = layers_enc

        self.var_z_clean_encoder = baseEncoder(
            dataset_obj, args, _enc_z_layers, self.activ
        )

        self.var_z_dirty_encoder = baseEncoder(
            dataset_obj, args, _enc_z_layers, self.activ
        )

        # z_clean
        self._mu_z_clean = modSeq(
            [
                self.var_z_clean_encoder,
                nn.Linear(
                    self.var_z_clean_encoder.layers_list[-1][1], self.z_clean_dim
                ),
            ]
        )
        self._logvar_z_clean = modSeq(
            [
                self.var_z_clean_encoder,
                nn.Linear(
                    self.var_z_clean_encoder.layers_list[-1][1], self.z_clean_dim
                ),
            ]
        )
        self._z_clean_params = {"mu": self._mu_z_clean, "logvar": self._logvar_z_clean}
        self.var_q_z_clean = GaussDiagDistModule(
            self.z_clean_dim,
            plugin_params=True,
        )

        # z_dirty
        self._mu_z_dirty = modSeq(
            [
                self.var_z_dirty_encoder,
                nn.Linear(
                    self.var_z_dirty_encoder.layers_list[-1][1], self.z_dirty_dim
                ),
            ]
        )
        self._logvar_z_dirty = modSeq(
            [
                self.var_z_dirty_encoder,
                nn.Linear(
                    self.var_z_dirty_encoder.layers_list[-1][1], self.z_dirty_dim
                ),
            ]
        )
        self._z_dirty_params = {"mu": self._mu_z_dirty, "logvar": self._logvar_z_dirty}
        self.var_q_z_dirty = GaussDiagDistModule(
            self.z_dirty_dim,
            plugin_params=True,
        )

        # p(z_\epsilon) or p(z_eps) -- gaussian noise, used in latent space
        _sigma_eps = args.sigma_eps_z_in
        _static_params_eps = {
            "mu": args.mean_eps_z_in * torch.ones(self.z_dirty_dim),
            "logvar": ((_sigma_eps ** 2) * torch.ones(self.z_dirty_dim)).log(),
        }
        self.var_eps_noise = GaussDiagDistModule(
            self.z_dirty_dim, static_params=_static_params_eps
        )

        # p(y) -- prior on ratio of inliers in the data
        _static_prior_y = torch.tensor(args.y_clean_prior)
        self.prior_y = BernoulliDistModule(static_prob=_static_prior_y)

        # p(z_clean) or p(z_c) -- clean representation for underlying inlier
        _static_prior_z_clean = {
            "mu": torch.zeros(self.z_clean_dim),
            "logvar": (
                (args.fixed_prior_z_clean ** 2) * torch.ones(self.z_clean_dim)
            ).log(),
        }
        self.prior_z_clean = GaussDiagDistModule(
            self.z_clean_dim, static_params=_static_prior_z_clean
        )

        # p(z_dirty) or p(z_d) -- dirty representation for the error
        _static_prior_z_dirty = {
            "mu": torch.zeros(self.z_dirty_dim),
            "logvar": (
                (args.fixed_prior_z_dirty ** 2) * torch.ones(self.z_dirty_dim)
            ).log(),
        }
        self.prior_z_dirty = GaussDiagDistModule(
            self.z_dirty_dim, static_params=_static_prior_z_dirty
        )

        # p(x|z) or p(x| z_c, z_d, z_eps, y) -- decoder
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
            "dist_corr",
        ]

    def get_z_dists(self):
        """ Used for its r_sample and log_q functions alone """

        mdl_dists = dict()

        mdl_dists["q_z_c"] = GaussDiagDistModule(self.z_clean_dim)
        mdl_dists["p_z_c"] = self.prior_z_clean

        mdl_dists["q_z_d"] = GaussDiagDistModule(self.z_dirty_dim)
        mdl_dists["p_z_d"] = self.prior_z_dirty

        return mdl_dists

    def forward(self, x_data, n_epoch=None, y_targets=None, repair_mode=False):

        # p_params --> reconstruction (recon) params (softmax probs and gaussian per feature), and prior params
        # q_params --> mu, logvar of gauss variational dist
        # q_samples --> samples from gauss variational dist

        _device_used = x_data.device
        _len_batch = x_data.shape[0]

        q_params = dict()
        q_samples = dict()
        log_q_dists = dict()

        # encode to obtain params for z_clean (or z_c)
        z_clean_base_enc = self.var_z_clean_encoder(x_data)
        _enc_param_dict_clean = {
            "mu": self._mu_z_clean.mods_list[-1](z_clean_base_enc),
            "logvar": self._logvar_z_clean.mods_list[-1](z_clean_base_enc),
        }

        # encode to obtain params for z_dirty (or z_d)
        z_dirty_base_enc = self.var_z_dirty_encoder(x_data)
        _enc_param_dict_dirty = {
            "mu": self._mu_z_dirty.mods_list[-1](z_dirty_base_enc),
            "logvar": self._logvar_z_dirty.mods_list[-1](z_dirty_base_enc),
        }

        self.var_q_z_clean.update_params_manual(_enc_param_dict_clean)
        self.var_q_z_dirty.update_params_manual(_enc_param_dict_dirty)

        q_params["z_clean"], q_samples["z_clean"], _ = self.var_q_z_clean(
            sampling=True, evalprob=False
        )

        q_params["z_dirty"], q_samples["z_dirty"], _ = self.var_q_z_dirty(
            sampling=True, evalprob=False
        )

        # z_\epsilon, get noise samples and parameters
        q_params["eps_noise"], _, _ = self.var_eps_noise()
        q_params["eps_noise"]["mu"] = (
            q_params["eps_noise"]["mu"]
            .unsqueeze(0)
            .expand(_len_batch, self.z_dirty_dim)
        )
        q_params["eps_noise"]["logvar"] = (
            q_params["eps_noise"]["logvar"]
            .unsqueeze(0)
            .expand(_len_batch, self.z_dirty_dim)
        )

        q_samples["eps_noise"] = self.var_eps_noise.r_samples(n_samples=_len_batch)
        q_samples["eps_noise"] = q_samples["eps_noise"].squeeze(0)

        # clean representation z | y=1 (lower manifold): concat [z_c ; z_\epsilon]
        q_params["z_y1"] = dict()
        q_params["z_y1"]["mu"] = torch.cat(
            [q_params["z_clean"]["mu"], q_params["eps_noise"]["mu"]], 1
        )
        q_params["z_y1"]["logvar"] = torch.cat(
            [q_params["z_clean"]["logvar"], q_params["eps_noise"]["logvar"]], 1
        )

        q_samples["z_y1"] = torch.cat([q_samples["z_clean"], q_samples["eps_noise"]], 1)

        # dirty representation z | y=0 (higher manifold): concat [z_c ; z_d]
        q_params["z_y0"] = dict()

        q_params["z_y0"]["mu"] = torch.cat(
            [q_params["z_clean"]["mu"], q_params["z_dirty"]["mu"]], 1
        )
        q_params["z_y0"]["logvar"] = torch.cat(
            [q_params["z_clean"]["logvar"], q_params["z_dirty"]["logvar"]], 1
        )
        q_samples["z_y0"] = torch.cat([q_samples["z_clean"], q_samples["z_dirty"]], 1)

        p_params = dict()
        p_params["y"], _, _ = self.prior_y()
        p_params["z_clean"], _, _ = self.prior_z_clean()
        p_params["z_dirty"], _, _ = self.prior_z_dirty()

        # q(y|z)
        _q_samples_z_y0 = torch.cat(
            [q_samples["z_clean"].detach(), q_samples["z_dirty"]], 1
        ) # NOTE: stop gradient (i.e. detach) is used in z_clean (or z_c)
        q_params["y"], _, _ = self.q_y_dist(
            _q_samples_z_y0, sampling=False, evalprob=False
        )

        # NOTE: to be used in: supervised part of loss; and outlier detection (OD) analytics.
        if y_targets is None:
            log_q_dists["y"] = []
        else:
            _y_targets = y_targets.float()
            log_q_dists["y"] = self.q_y_dist.log_probs(
                _y_targets,
                force_params=q_params["y"],
                weights=self.args.qy_sup_weights,
            )

        p_params["recon_y1"] = self.decoder(q_samples["z_y1"])
        p_params["recon_y0"] = self.decoder(q_samples["z_y0"])

        # NOTE: to be used in eval / repair; uses MAP solution from clean component.
        q_params["z"] = q_params["z_y1"]
        q_samples["z"] = q_samples["z_y1"]
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
        reg_scheduler_val=1.0,
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

        ret_list = semi_y_vae_partitioned_ELBO(
            self.dataset_obj,
            input_data,
            p_params,
            q_params,
            y_targets=_y_targets,
            mask_semi=_mask_semi,
            kl_coeff=kl_coeff,
            q_samples=q_samples,
            dist_corr_reg=self.args.dist_corr_reg,
            dist_corr_reg_coeff=self.args.dist_corr_reg_coeff,
            reg_scheduler_val=reg_scheduler_val,
            n_epoch=n_epoch,
        )

        (
            elbo_loss,
            nll,
            kld_tot,
            kld_y,
            kld_z_y1,
            kld_z_y0,
            dist_corr,
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

        # total_loss = elbo_loss + sup_coeff * sup_loss
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
                ("dist_corr", dist_corr),
            ]
        )

        return ret_dict
