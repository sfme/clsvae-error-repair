import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from collections import OrderedDict
from repair_syserr_models.model_utils import (
    logit_fn,
    trf_param_to_cholesky,
    masker,
    repeat_rows,
    split_leading_dim,
)

EPS = 1e-8


class modSeq(nn.Module):
    def __init__(self, mods_list):

        super().__init__()

        self.mods_list = nn.ModuleList(mods_list)

    def forward(self, context_data, add_in=None, in_dropout=False):

        nn_fargs = [context_data, add_in, in_dropout]
        push_h = self.mods_list[0](*nn_fargs)

        for mod_obj in self.mods_list[1:]:
            push_h = mod_obj(push_h)  # __call__()

        return push_h


class reshapeToMatrix(nn.Module):
    def __init__(self, x_dim, y_dim):

        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

    def forward(self, in_):
        return in_.view(in_.shape[0], self.x_dim, self.y_dim)


class baseEncoder(nn.Module):
    def __init__(
        self, dataset_obj, args, layers_list, activ, add_in_len=0, in_p_dropout=0.0
    ):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        self.len_in_feats = len(dataset_obj.feat_info)

        _layers_list = [list(tup) for tup in layers_list]
        _layers_list[0][0] = self.args.size_input + add_in_len

        self.in_p_dropout = in_p_dropout  # input feature dropout

        self.activ = activ
        self.layers_list = _layers_list

        # Encoder Params

        # define a different embedding matrix for each categorical feature
        if dataset_obj.dataset_type == "image":
            self.feat_embedd = nn.ModuleList([])
        else:
            self.feat_embedd = nn.ModuleList(
                [
                    nn.Embedding(c_size, self.args.embedding_size, max_norm=1)
                    for _, col_type, c_size in dataset_obj.feat_info
                    if col_type == "categ"
                ]
            )

        net_build = []
        for lay_idx, (layer_in, layer_out) in enumerate(_layers_list):
            net_build.append(("fc_" + str(lay_idx), nn.Linear(layer_in, layer_out)))
            net_build.append(("activ_" + str(lay_idx), self.activ))

            if self.args.use_batch_norm:
                net_build.append(
                    ("norm_layer_" + str(lay_idx), nn.BatchNorm1d(layer_out))
                )

        self.mod_fw_seq = nn.Sequential(OrderedDict(net_build))


    def get_inputs(self, x_data, drop_mask=None):

        _len_batch = x_data.shape[0]

        if not drop_mask:
            drop_mask = torch.ones(
                (_len_batch, self.len_in_feats), device=x_data.device
            )

        if self.dataset_obj.dataset_type == "image":
            # image data: for now assume real features, or binary value features.
            return x_data * drop_mask

        else:
            # mixed data, or just real or just categ
            input_list = []
            cursor_embed = 0

            for feat_idx, (_, col_type, feat_size) in enumerate(
                self.dataset_obj.feat_info
            ):

                if col_type == "categ":  # categorical (uses embeddings)
                    _aux_categ = self.feat_embedd[cursor_embed](
                        x_data[:, feat_idx].long()
                    )
                    _aux_categ = _aux_categ * drop_mask[:, feat_idx].view(-1, 1)
                    input_list.append(_aux_categ)
                    cursor_embed += 1

                elif col_type == "real":  # numerical
                    input_list.append(
                        x_data[:, feat_idx].view(-1, 1)
                        * drop_mask[:, feat_idx].view(-1, 1)
                    )

            return torch.cat(input_list, 1)

    def forward(self, x_data, add_in=None, in_dropout=False):

        if in_dropout:
            _shape = (x_data.shape[0], self.len_in_feats)
            _logit_drop = logit_fn(self.in_p_dropout)
            _mask_drop = masker(_logit_drop, dropout=True, out_shape=_shape)
        else:
            _mask_drop = None

        if add_in is None:
            push_fw = self.get_inputs(x_data, _mask_drop)
        else:
            push_fw = torch.cat([self.get_inputs(x_data, _mask_drop), add_in], dim=1)

        return self.mod_fw_seq(push_fw)


class baseDecoder(nn.Module):
    def __init__(self, dataset_obj, args, layers_list, activ):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        self.activ = activ
        self.layers_list = layers_list

        # Decoder Params

        net_build = []
        for lay_idx, (layer_in, layer_out) in enumerate(layers_list):
            net_build.append(("fc_" + str(lay_idx), nn.Linear(layer_in, layer_out)))
            net_build.append(("activ_" + str(lay_idx), self.activ))

            if self.args.use_batch_norm:
                net_build.append(
                    ("norm_layer_" + str(lay_idx), nn.BatchNorm1d(layer_out))
                )

        self.mod_fw_seq = nn.Sequential(OrderedDict(net_build))

        if dataset_obj.dataset_type == "image":
            self.out_cat_linears = nn.Linear(layers_list[-1][1], self.args.size_output)
        else:
            self.out_cat_linears = nn.ModuleList(
                [
                    nn.Linear(layers_list[-1][1], c_size)
                    if col_type == "categ"
                    else nn.Linear(layers_list[-1][1], c_size)  # 2*
                    for _, col_type, c_size in dataset_obj.feat_info
                ]
            )

        ## Log variance of the decoder for real attributes
        if dataset_obj.dataset_type == "image":
            if args.use_binary_img:
                self.logvar_x = []
            else:
                self.logvar_x = nn.Parameter(torch.zeros(1).float())
        else:
            if dataset_obj.num_cols:
                self.logvar_x = nn.Parameter(
                    torch.zeros(1, len(dataset_obj.num_cols)).float()
                )
            else:
                self.logvar_x = []

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z):

        recon_params = dict()
        h_final = self.mod_fw_seq(z)

        if self.dataset_obj.dataset_type == "image":
            # tensor with dims (batch_size, self.size_output)
            if self.args.use_binary_img:
                recon_params["x"] = torch.sigmoid(self.out_cat_linears(h_final))
            else:
                recon_params["x"] = self.out_cat_linears(h_final)
                recon_params["logvar_x"] = self.logvar_x.clamp(-10, 3)  # -3, 3;

        else:
            out_cat_list = []
            for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):

                if self.dataset_obj.feat_info[feat_idx][1] == "categ":  # coltype check
                    out_cat_list.append(self.logSoftmax(out_cat_layer(h_final)))

                elif self.dataset_obj.feat_info[feat_idx][1] == "real":
                    out_cat_list.append(out_cat_layer(h_final))

            # tensor with dims (batch_size, self.size_output)
            recon_params["x"] = torch.cat(out_cat_list, 1)

            if self.dataset_obj.num_cols:
                recon_params["logvar_x"] = self.logvar_x.clamp(-10, 3)  # -3, 3;
                # recon_params["logvar_x"] = torch.zeros_like(self.logvar_x)

        return recon_params


class encodeCateg(nn.Module):

    # neural net encoder for categorical distribution

    def __init__(self, dataset_obj, args, n_comps, mod_torso_enc=None, **enc_args):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        self.n_comps = n_comps

        if mod_torso_enc is None:
            if "add_in_len" not in enc_args:
                enc_args["add_in_len"] = 0
            if "in_p_dropout" not in enc_args:
                enc_args["in_p_dropout"] = 0.0

            self.torso_encoder = baseEncoder(
                dataset_obj,
                args,
                enc_args["layers_list"],
                enc_args["activ"],
                enc_args["add_in_len"],
                enc_args["in_p_dropout"],
            )
        else:
            self.torso_encoder = mod_torso_enc

        aux_list = [
            self.torso_encoder,
            nn.Linear(self.torso_encoder.layers_list[-1][1], self.n_comps),
        ]
        logit_net = modSeq(aux_list)

        self.param_nn = CategDistModule(
            self.n_comps, temp_gs=0.7, func_nn={"logits": logit_net}
        )  
        # TODO: put temp_gs flag from args? or in r_samples?

    def forward(
        self, input_data, add_in=None, sampling=False, evalprob=False, in_dropout=False
    ):

        return self.param_nn(input_data, add_in, sampling, evalprob, in_dropout)


class encodeMVNDiag(nn.Module):

    # simple MV. Gaussian Diag encoder (neural net), only one component

    def __init__(self, dataset_obj, args, var_dim, mod_torso_enc=None, **enc_args):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        self.var_dim = var_dim

        if mod_torso_enc is None:
            if "add_in_len" not in enc_args:
                enc_args["add_in_len"] = 0
            if "in_p_dropout" not in enc_args:
                enc_args["in_p_dropout"] = 0.0
            
            self.torso_encoder = baseEncoder(
                dataset_obj,
                args,
                enc_args["layers_list"],
                enc_args["activ"],
                enc_args["add_in_len"],
                enc_args["in_p_dropout"],
            )
        else:
            self.torso_encoder = mod_torso_enc

        aux_list = [
            self.torso_encoder,
            nn.Linear(self.torso_encoder.layers_list[-1][1], self.var_dim),
        ]
        mu_net = modSeq(aux_list)

        aux_list = [
            self.torso_encoder,
            nn.Linear(self.torso_encoder.layers_list[-1][1], self.var_dim),
        ]
        logvar_net = modSeq(aux_list)

        self.param_nn = GaussDiagDistModule(
            self.var_dim, {"mu": mu_net, "logvar": logvar_net}
        )

    def forward(
        self, input_data, add_in=None, sampling=False, evalprob=False, in_dropout=False
    ):

        return self.param_nn(input_data, add_in, sampling, evalprob, in_dropout)


class encodeBern(nn.Module):

    # neural network encoder for a bernoulli distribution

    def __init__(self, dataset_obj, args, mod_torso_enc=None, **enc_args):

        super().__init__()

        self.dataset_obj = dataset_obj
        self.args = args

        if mod_torso_enc is None:
            if "add_in_len" not in enc_args:
                enc_args["add_in_len"] = 0
            if "in_p_dropout" not in enc_args:
                enc_args["in_p_dropout"] = 0.0
            
            self.torso_encoder = baseEncoder(
                dataset_obj,
                args,
                enc_args["layers_list"],
                enc_args["activ"],
                enc_args["add_in_len"],
                enc_args["in_p_dropout"],
            )
        else:
            self.torso_encoder = mod_torso_enc

        aux_list = [
            self.torso_encoder,
            nn.Linear(self.torso_encoder.layers_list[-1][1], 1),
        ]
        logits_net = modSeq(aux_list)

        self.param_nn = BernoulliDistModule(temp_gs=0.7, func_nn={"logits": logits_net})
        # TODO: put temp_gs flag from args? or in r_samples? for flexibility.

    def forward(
        self, input_data, add_in=None, sampling=False, evalprob=False, in_dropout=False
    ):

        return self.param_nn(input_data, add_in, sampling, evalprob, in_dropout)


############# Distribution Modules


class GaussDiagDistModule(nn.Module):

    # Diagonal Multivariate Gaussian distribution module

    def __init__(
        self,
        latent_dim,
        func_nn=None,
        static_params=None,
        share_params=None,
        plugin_params=False,
    ):

        super().__init__()

        self.latent_dim = latent_dim

        if plugin_params:
            # if updating dynamically at train / test
            self.nn_used = False

        else:
            if static_params is None:
                if func_nn is None:
                    self.nn_used = False

                    if share_params is None:
                        self.mu = nn.Parameter(
                            torch.nn.init.normal_(torch.empty(self.latent_dim))
                        )
                        self.logvar = nn.Parameter(
                            torch.nn.init.normal_(torch.empty(self.latent_dim))
                        )
                    else:
                        self.mu = nn.Parameter(share_params["mu"])
                        self.logvar = nn.Parameter(share_params["logvar"])

                else:
                    self.nn_used = True
                    self.mu = func_nn["mu"]
                    self.logvar = func_nn["logvar"]

            else:
                self.nn_used = False
                self.register_buffer("mu", static_params["mu"])
                self.register_buffer("logvar", static_params["logvar"])

        self.params_nn_buffer = []

    def update_params_manual(self, in_params_dict):
        """
        - Used for dynamic update (forcing) at train or test time
        - NOTE: need to use plugin_params option in __init__
        - NOTE: needs to be called before at e.g. forward-pass, before sampling or log_prob func calls!!
        """

        self.mu = in_params_dict["mu"]  # BxD
        self.logvar = in_params_dict["logvar"]  # BxD

    def get_params(self, context_data=None, add_in=None, in_dropout=False):

        params = dict()

        if self.nn_used:
            if context_data is None:
                raise ValueError("Needs context vars or data as argument")
            nn_fargs = [context_data, add_in, in_dropout]
            params["mu"] = self.mu(*nn_fargs)
            params["logvar"] = self.logvar(*nn_fargs)
            self.params_nn_buffer = [params]
        else:
            params["mu"] = self.mu
            params["logvar"] = self.logvar

        return params

    def clear_param_buffer(self):

        self.params_nn_buffer = []

    def r_samples(
        self,
        context_data=None,
        add_in=None,
        force_params=None,
        in_dropout=False,
        n_samples=1,
    ):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in, in_dropout)
        else:
            params = force_params

        _mu = params["mu"].view(-1, self.latent_dim)  # BxD
        _logvar = params["logvar"].view(-1, self.latent_dim)  # BxD

        _len_batch = _mu.shape[0]  # B (can be 1)

        if n_samples > 1:
            _mu = repeat_rows(_mu, n_samples)  # (BxS)xD
            _logvar = repeat_rows(_logvar, n_samples)  # (BxS)xD

        # specific sampling
        if self.training:
            eps = torch.randn_like(_mu)
            std = _logvar.mul(0.5).exp_()
            _samples = eps.mul(std).add_(_mu)
        else:  # TODO: maybe should always be sampled / noised?
            _samples = _mu

        if n_samples > 1:
            return split_leading_dim(_samples, (_len_batch, n_samples))  # BxSxD
        else:
            return _samples  # BxD

    def log_probs(
        self,
        data_points,
        context_data=None,
        add_in=None,
        force_params=None,
        in_dropout=False,
    ):

        # input data_points size: BxD or 1xD or FxD, where F>1. D is latent_dim

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in, in_dropout)
        else:
            params = force_params

        # get log densities
        _mu = params["mu"].view(-1, self.latent_dim)
        _logvar = params["logvar"].view(-1, self.latent_dim)
        _data_points = data_points.view(-1, self.latent_dim)

        log_dens = ((_data_points - _mu) ** 2 / (_logvar.exp() + EPS)).sum(dim=-1)
        log_dens += _logvar.sum(dim=-1)
        log_dens += np.log(2 * np.pi) * self.latent_dim
        log_dens = -0.5 * log_dens

        return log_dens  # view() for Bx1 ?

    def forward(
        self,
        context_data=None,
        add_in=None,
        sampling=False,
        evalprob=False,
        clear_buffer=True,
        in_dropout=False,
    ):

        params = self.get_params(context_data, add_in, in_dropout)
        samples = self.r_samples() if sampling else []
        log_p = self.log_probs(samples) if evalprob else []

        if clear_buffer:
            self.params_nn_buffer = []

        return params, samples, log_p


class CategDistModule(nn.Module):

    # Categorical distribution module

    def __init__(self, n_comps, temp_gs=0.5, func_nn=None, static_params=None):

        super().__init__()

        self.n_comps = n_comps
        self.temp_gs = temp_gs

        if static_params is None:
            if func_nn is None:
                self.nn_used = False
                self.logits = nn.Parameter(
                    torch.nn.init.normal_(torch.empty(self.n_comps))
                )
            else:
                self.nn_used = True
                self.logits = func_nn["logits"]
        else:
            self.nn_used = False
            self.register_buffer("logits", static_params["logits"])

        self.params_nn_buffer = []

    def get_params(self, context_data=None, add_in=None, in_dropout=False):

        params = dict()

        if self.nn_used:
            if context_data is None:
                raise ValueError("Needs context vars or data as argument")
            nn_fargs = [context_data, add_in, in_dropout]
            params["logits"] = self.logits(*nn_fargs)
            self.params_nn_buffer = [params]
        else:
            params["logits"] = self.logits

        return params

    def clear_param_buffer(self):

        self.params_nn_buffer = []

    def r_samples(
        self, context_data=None, add_in=None, force_params=None, in_dropout=False
    ):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in, in_dropout)
        else:
            params = force_params

        # specific sampling (relaxed categorical: gumbel-softmax)
        if self.training:
            u_samps = torch.rand(
                params["logits"].shape,
                dtype=params["logits"].dtype,
                device=params["logits"].device,
            )
            u_samps = torch.clamp(u_samps, EPS, 1.0 - EPS)
            gumbels = -((-(u_samps.log())).log())
        else:  # TODO: maybe should always be sampled / noised?
            gumbels = 0.0

        score = (params["logits"] + gumbels) / self.temp_gs
        score = score - score.logsumexp(dim=-1, keepdim=True)

        return score.exp()

    def log_probs(
        self,
        data_points,
        context_data=None,
        add_in=None,
        force_params=None,
        in_dropout=False,
    ):

        # data points size: BxK; FxK, where is >1. They are one-hot vectors across dim K.

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in, in_dropout)
        else:
            params = force_params

        # get log probabilities
        _logits = params["logits"].view(-1, self.n_comps)
        _data_points = data_points.view(-1, self.n_comps)

        log_probs = F.log_softmax(_logits, dim=-1)
        log_like = (_data_points * log_probs).sum(dim=-1)

        return log_like

    def forward(
        self,
        context_data=None,
        add_in=None,
        sampling=False,
        evalprob=False,
        clear_buffer=True,
        in_dropout=False,
    ):

        params = self.get_params(context_data, add_in, in_dropout)
        samples = self.r_samples() if sampling else []
        log_p = self.log_probs(samples) if evalprob else []

        if clear_buffer:
            self.params_nn_buffer = []

        return params, samples, log_p


class GMMDistModule(nn.Module):
    """
    GMM distribution module

    - uses diagonal covariance components
    - number of components defined by user
    - usually used as prior dist. for latent space (VAE)
    """

    def __init__(
        self,
        n_comps,
        latent_dim,
        func_nn_categ=None,
        func_nn_comps=None,
        share_param_comps=None,
        static_param_comps=None,
        static_param_categ=None,
    ):

        super().__init__()

        self.n_comps = n_comps  # K
        self.latent_dim = latent_dim  # D

        if func_nn_comps is None:
            self.nn_used = False
            if share_param_comps is None:
                if static_param_comps is None:
                    # create params
                    self.mu = nn.Parameter(
                        torch.nn.init.normal_(
                            torch.empty(self.n_comps, self.latent_dim)
                        )
                    )  # KxD
                    self.logvar = nn.Parameter(
                        torch.nn.init.normal_(
                            torch.empty(self.n_comps, self.latent_dim)
                        )
                    )  # KxD
                else:
                    self.register_buffer("mu", static_param_comps["mu"])
                    self.register_buffer("logvar", static_param_comps["logvar"])
            else:
                self.mu = share_param_comps["mu"]
                self.logvar = share_param_comps["logvar"]
        else:
            self.nn_used = True
            self.mu = func_nn_comps["mu"]  # KxD 
            self.logvar = func_nn_comps["logvar"]  # KxD

        self.mix_categ = CategDistModule(
            self.n_comps,
            func_nn=func_nn_categ,
            static_params=static_param_categ,
            temp_gs=0.5,
        )
        # NOTE: temp_gs, only needed for r_samples in compute graph

        self.params_nn_buffer = []

    def get_params(self, context_data=None, add_in=None):

        params = dict()

        if self.nn_used:
            if context_data is None:
                raise ValueError("Needs context vars or data as argument")
            if add_in is None:
                nn_fargs = [context_data]
            else:
                nn_fargs = [context_data, add_in]
            params["logits"] = self.mix_categ.logits(*nn_fargs)
            params["mu"] = self.mu(*nn_fargs)
            params["logvar"] = self.logvar(*nn_fargs)
            self.params_nn_buffer = [params]
        else:
            params["logits"] = self.mix_categ.logits
            params["mu"] = self.mu
            params["logvar"] = self.logvar

        return params

    def clear_param_buffer(self):

        self.params_nn_buffer = []

    def r_samples(self, context_data=None, add_in=None, force_params=None):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            params = force_params

        ## gumbel-softmax trick; used in for categorical dist. of the mixture
        u_samps = torch.rand(
            params["logits"].shape, dtype=params["logits"].dtype
        )  # BxK or FxK
        u_samps = torch.clamp(u_samps, EPS, 1.0 - EPS)
        gumbels = -((-(u_samps.log())).log())

        score = (params["logits"] + gumbels) / self.temp_gs
        mix_select_soft = (score - score.logsumexp(dim=-1, keepdim=True)).exp()

        ## hard selection (needed to pick a component per sample)
        index = mix_select_soft.max(dim=-1, keepdim=True)[1]
        mix_select_hard = torch.zeros_like(mix_select_soft).scatter_(-1, index, 1.0)

        mask_selects = (
            mix_select_hard - mix_select_soft.detach() + mix_select_soft
        )  # needed if used in bprop with hard selects

        ## use selected component, in mask_selects, and sample that Gaussian dist.
        # NOTE: params['mu'] or params['logvar'] is BxKxD
        _masked_mu = (mask_selects.view(-1, self.n_comps, 1) * params["mu"]).sum(
            dim=1
        )  # BxD
        _masked_logvar = (
            mask_selects.view(-1, self.n_comps, 1) * params["logvar"]
        ).sum(
            dim=1
        )  # BxD

        eps = torch.randn_like(_masked_mu)
        std = _masked_logvar.mul(0.5).exp_()
        final_mix_samples = eps.mul(std).add_(_masked_mu)  # BxD; B samples of size D

        return final_mix_samples  # TODO: return assignments 'a' (GS samples) as well?

    def log_probs(self, data_points, context_data=None, add_in=None, force_params=None):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            params = force_params

        _data_points = data_points.view(
            -1, 1, self.latent_dim
        )  # NOTE: unsqueeze(.,-2)?

        # get log_p's of categorical dist
        _logits = params["logits"].view(-1, self.n_comps)
        log_probs = F.log_softmax(_logits, dim=-1)  # BxK

        # get log_p's of gaussian diagonal dists
        _mu = params["mu"].view(-1, self.n_comps, self.latent_dim)
        _logvar = params["logvar"].view(-1, self.n_comps, self.latent_dim)

        log_dens = ((_data_points - _mu) ** 2 / (_logvar.exp() + EPS)).sum(
            dim=-1
        )  # BxK
        log_dens += _logvar.sum(dim=-1)  # BxK
        log_dens += np.log(2 * np.pi) * self.latent_dim
        log_dens = -0.5 * log_dens  # BxK

        # get mixture log_p
        mix_log_p = (log_probs * log_dens).sum(dim=-1)  # B

        return mix_log_p

    def forward(
        self,
        context_data=None,
        add_in=None,
        sampling=False,
        evalprob=False,
        clear_buffer=True,
    ):

        params = self.get_params(context_data, add_in)
        samples = self.r_samples() if sampling else []
        log_p = self.log_probs(samples) if evalprob else []

        if clear_buffer:
            self.params_nn_buffer = []

        return params, samples, log_p


class BernoulliDistModule(nn.Module):

    # NOTE: can use -log_probs as NLL in training semi-supervised.

    def __init__(self, temp_gs=0.5, func_nn=None, static_prob=None):

        super().__init__()

        self.temp_gs = temp_gs

        if static_prob is None:
            if func_nn is None:
                self.nn_used = False
                self.logits = nn.Parameter(torch.nn.init.normal_(torch.empty(1)))
            else:
                self.nn_used = True
                self.logits = func_nn["logits"]
        else:
            # NOTE: if static, provides self.register_buffer(...) already.
            self.nn_used = False
            self.register_buffer("logits", logit_fn(static_prob))

        self.params_nn_buffer = []

    def get_params(self, context_data=None, add_in=None):

        params = dict()

        if self.nn_used:
            if context_data is None:
                raise ValueError("Needs context vars or data as argument")
            if add_in is None:
                nn_fargs = [context_data]
            else:
                nn_fargs = [context_data, add_in]
            params["logits"] = self.logits(*nn_fargs)
            self.params_nn_buffer = [params]
        else:
            params["logits"] = self.logits

        return params

    def clear_param_buffer(self):

        self.params_nn_buffer = []

    def log_probs(
        self,
        data_points,
        context_data=None,
        add_in=None,
        force_params=None,
        weights=None,
    ):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (
                context_data is None
            ):  # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            params = force_params

        if weights is None:
            _weights = [1.0, 1.0]
        else:
            _weights = weights

        _probs = torch.sigmoid(params["logits"]).view(-1, 1)

        _data_points = data_points.view(-1, 1)

        log_like = _weights[0] * _data_points * torch.log(_probs + EPS)
        log_like = log_like + _weights[1] * (1.0 - _data_points) * torch.log(
            1.0 - _probs + EPS
        )

        return log_like

    def r_samples(self, context_data=None, add_in=None, force_params=None):

        # Bx1 or Fx1;

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (
                context_data is None
            ):  # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            params = force_params

        noise = torch.rand_like(params["logits"])
        noise = logit_fn(noise)
        logit_samples = (params["logits"] + noise) / self.temp_gs

        if self.training:
            # returns soft binary w's (training)
            sample_ret = torch.sigmoid(logit_samples)
        else:
            # returns hard binary w's (at test time)
            sample_ret = torch.round(torch.sigmoid(logit_samples))

        return sample_ret

    def forward(
        self,
        context_data=None,
        add_in=None,
        sampling=False,
        evalprob=False,
        clear_buffer=True,
    ):

        params = self.get_params(context_data, add_in)
        samples = self.r_samples() if sampling else []
        log_p = self.log_probs(samples) if evalprob else []

        if clear_buffer:
            self.params_nn_buffer = []

        return params, samples, log_p


class GaussFullCovDistModule(nn.Module):

    # Full Covariance Multivariate Gaussian

    # Parameters: mu (mean vector), covar_chol (cholesky factor of covar matrix, lower triangular)

    def __init__(
        self,
        latent_dim,
        apply_trf_chol=True,
        func_nn=None,
        static_params=None,
        share_params=None,
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.apply_trf_chol = apply_trf_chol

        if static_params is None:
            if func_nn is None:
                self.nn_used = False

                if share_params is None:
                    self.mu = nn.Parameter(
                        torch.nn.init.normal_(torch.empty(self.latent_dim))
                    )
                    n_elems_chol = int(latent_dim * (latent_dim + 1) / 2)
                    self.covar_chol = nn.Parameter(
                        torch.nn.init.normal_(torch.empty(n_elems_chol))
                    )
                    self.apply_trf_chol = True

                else:
                    self.mu = nn.Parameter(share_params["mu"])
                    self.covar_chol = nn.Parameter(share_params["covar_chol"])
                    # self.apply_trf_chol --> see default (True)

            else:
                self.nn_used = True
                self.mu = func_nn["mu"]
                self.covar_chol = func_nn["covar_chol"]
                # self.apply_trf_chol --> see default (True)

        else:
            self.nn_used = False
            self.register_buffer("mu", static_params["mu"])
            self.register_buffer("covar_chol", static_params["covar_chol"])
            self.apply_trf_chol = False

        self.params_nn_buffer = []

    def _trf_get_chol(self, param_tensor):

        """
        inputs: param_tensor (BxN or 1xN) is batch tensor where 2nd dim holds cholesky matrix
        entries. The first 'latent_dim' elements in 2nd dim are diagonal entries
        (i.e. >0), the rest are lower triangular matrix elements (below diagonal).

        NOTE: number of non-zero cholesky entries is n*(n+1)/2

        outputs: 1xDxD or BxDxD

        NOTE: Calls external function, from model_utils.py.
        """

        return trf_param_to_cholesky(self.latent_dim, param_tensor)

    def get_params(self, context_data=None, add_in=None):

        params = dict()

        if self.nn_used:
            if context_data is None:
                raise ValueError("Needs context vars or data as argument")
            if add_in is None:
                nn_fargs = [context_data]
            else:
                nn_fargs = [context_data, add_in]

            params["mu"] = self.mu(*nn_fargs)
            if self.apply_trf_chol:
                params["covar_chol"] = self._trf_get_chol(self.covar_chol(*nn_fargs))
            else:
                params["covar_chol"] = self.covar_chol(*nn_fargs)

            self.params_nn_buffer = [params]
        else:
            params["mu"] = self.mu
            if self.apply_trf_chol:
                params["covar_chol"] = self._trf_get_chol(self.covar_chol)
            else:
                params["covar_chol"] = self.covar_chol

        return params

    def clear_param_buffer(self):

        self.params_nn_buffer = []

    def r_samples(self, context_data=None, add_in=None, force_params=None):

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            _shape_covar = force_params["covar_chol"].shape
            assert _shape_covar[-1] == _shape_covar[-2]
            assert _shape_covar[-1] == self.latent_dim

            params = force_params

        _mu = params["mu"].view(-1, self.latent_dim)
        _covar_chol = params["covar_chol"].view(
            -1, self.latent_dim, self.latent_dim
        )  # lower triangular cholesky matrix

        if self.training:
            eps = torch.randn_like(_mu).unsqueeze(-1)
            return _mu + (_covar_chol @ eps).squeeze(-1)
        else:
            return _mu

    def _get_cov_inv(self, covar_chol):
        """
        input: lower cholesky decomp of covar matrix; BxDxD or 1xDxD
        returns: inverse of covar matrix; BxDxD or 1xDxD
        """

        batch_eye = torch.eye(covar_chol.shape[-1], device=covar_chol.device)
        batch_eye = batch_eye.repeat(covar_chol.shape[0], 1, 1)

        chol_inv = torch.triangular_solve(batch_eye, covar_chol, upper=False).solution
        covar_inv = chol_inv.transpose(-1, -2) @ chol_inv

        return covar_inv

    def _get_logdet_chol(self, covar_chol):
        """
        input: lower cholesky decomp of covar matrix
        returns: log det covar matrix, Bx1
        """

        diag_vecs = torch.diagonal(covar_chol, dim1=-2, dim2=-1)
        return 2 * diag_vecs.log().sum(dim=-1, keepdims=True)

    def log_probs(self, data_points, context_data=None, add_in=None, force_params=None):
        # input data_points size: BxD or 1xD or FxD, where F>1. D is latent_dim

        if force_params is None:
            if (context_data is None) and (not self.params_nn_buffer) and self.nn_used:
                raise ValueError(
                    "params need to be initialized for case of func() or NN"
                )

            if self.params_nn_buffer and (context_data is None):
                # only when self.nn_used == True
                params = self.params_nn_buffer[0]
            else:
                params = self.get_params(context_data, add_in)
        else:
            _shape_covar = force_params["covar_chol"].shape
            assert _shape_covar[-1] == _shape_covar[-2]
            assert _shape_covar[-1] == self.latent_dim

            params = force_params

        # get log densities
        _mu = params["mu"].view(-1, self.latent_dim)
        _covar_chol = params["covar_chol"].view(
            -1, self.latent_dim, self.latent_dim
        )  # lower triangular cholesky matrix
        _data_points = data_points.view(-1, self.latent_dim)

        cov_inv = self._get_cov_inv(_covar_chol)
        log_det_cov = self._get_logdet_chol(_covar_chol)

        dist_euc = (_data_points - _mu).unsqueeze(-1)
        mahal_dist = (dist_euc.transpose(-1, -2) @ cov_inv @ dist_euc).squeeze(-1)

        const_val = np.log(2 * np.pi) * self.latent_dim

        log_dens = -0.5 * (const_val + mahal_dist + log_det_cov)

        return log_dens

    def forward(
        self,
        context_data=None,
        add_in=None,
        sampling=False,
        evalprob=False,
        clear_buffer=True,
    ):

        params = self.get_params(context_data, add_in)
        samples = self.r_samples() if sampling else []
        log_p = self.log_probs(samples) if evalprob else []

        if clear_buffer:
            self.params_nn_buffer = []

        return params, samples, log_p

