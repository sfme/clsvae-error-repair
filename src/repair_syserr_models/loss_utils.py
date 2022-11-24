import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

from repair_syserr_models.model_utils import (
    repeat_rows,
    log_mean_exp,
    split_leading_dim,
    merge_leading_dims,
    logit_fn,
)

from repair_syserr_models.module_utils import GaussFullCovDistModule

EPS = 1e-8


##### KL Divs, NLLs, and Other Losses


def nll_categ_global(categ_logp_feat, input_idx_feat):

    return F.nll_loss(categ_logp_feat, input_idx_feat, reduction="none").view(-1, 1)


def nll_gauss_global(gauss_params, input_val_feat, logvar, shape_feats=[1]):

    mu = gauss_params.view([-1] + shape_feats)
    logvar_r = (logvar.exp() + 1e-9).log()

    data_compnt = 0.5 * logvar_r + (
        input_val_feat.view([-1] + shape_feats) - mu
    ) ** 2 / (2.0 * logvar_r.exp() + 1e-9)

    return data_compnt.view([-1] + shape_feats)


def nll_binary_global(logit_p, target):

    """ log_p is log of positive event probability (e.g. pixel_value=1) """

    # _p = torch.sigmoid(logit_p)
    # log_loss = target * torch.log(_p + EPS) + (1 - target) * torch.log(1 - _p + EPS)

    # return -log_loss

    return F.binary_cross_entropy_with_logits(logit_p, target, reduction="none")


def kl_gauss_diag(var_params, prior_params):

    # input size var_params: BxKxD or Bx1xD
    # input size prior_params: BxKxD or 1xKxD (e.g. prior that needs expanding)
    # output size: BxK (collapses on each component, sums over latent_dim)

    kld = (var_params["logvar"] - prior_params["logvar"]).exp()

    kld = kld + (prior_params["mu"] - var_params["mu"]) ** 2 / (
        prior_params["logvar"].exp() + EPS
    )

    kld = kld - 1 + (prior_params["logvar"] - var_params["logvar"])

    kld = 0.5 * kld

    return kld.sum(dim=-1)


def kl_gauss_full_cov(var_params, prior_params, is_prior_diag=False):

    dim_space = var_params["mu"].shape[-1]

    # NOTE: for prior, needs covar: 1xDxD or BxDxD; or logvar: 1xD or BxD;
    if is_prior_diag:  # SxD
        _inverse_prior = (-prior_params["logvar"]).exp()
        _inverse_prior = torch.diag_embed(_inverse_prior)
        _log_det_prior = prior_params["logvar"].sum(dim=-1)
    else:  # SxDxD
        _inverse_prior = torch.inverse(prior_params["covar"])  # takes in BxDxD or DxD
        # TODO: special structure for inverse? (less expensive)
        _log_det_prior = torch.logdet(prior_params["covar"])  # takes in BxDxD or DxD

    # trace of mat mult
    kld = torch.einsum("bii->b", _inverse_prior @ var_params["covar"]).view(-1, 1)

    # mahalanobis distance
    _delta_mus = prior_params["mu"] - var_params["mu"]
    _mahaldist = _delta_mus.unsqueeze(1) @ _inverse_prior @ _delta_mus.unsqueeze(-1)
    _mahaldist = _mahaldist.view(-1, 1)

    kld = kld + _mahaldist

    # logdet subtraction
    _logdet_delta = (_log_det_prior - torch.logdet(var_params["covar"])[1]).view(-1, 1)
    # TODO: special structure (previous spectral decomp.) for logdet of var_params['covar']? (less expensive)

    kld = 0.5 * (kld + _logdet_delta - dim_space)

    return kld  # rets Bx1


def kl_bern(var_logit, prior_logit):

    # input size: B (or Bx1)
    # output size: B (or Bx1)

    var_prob = torch.sigmoid(var_logit)
    prior_prob = torch.sigmoid(prior_logit)

    kld = var_prob * (torch.log(var_prob + EPS) - torch.log(prior_prob + EPS))
    kld = kld + (1.0 - var_prob) * (
        torch.log(1.0 - var_prob + EPS) - torch.log(1.0 - prior_prob + EPS)
    )

    return kld  # .view(-1,1)


def kl_categ(var_logits, prior_logits):

    # input size var_params: BxK
    # input size prior_params: 1xK (prior); or BxK if not prior
    # output size: B or Bx1

    varprobs = F.softmax(var_logits, dim=-1)
    priorprobs = F.softmax(prior_logits, dim=-1)

    kld = varprobs * (torch.log(varprobs + EPS) - torch.log(priorprobs + EPS))

    return kld.sum(dim=-1, keepdim=True)


def kl_gauss_diag_mixture(var_params, prior_params):

    # full analytical
    # input -- var: (categ) BxK or (gauss comps) BxKxD ; prior: (categ) K or (gauss comps) KxD
    # output size: Bx1

    # var_params and prior_params dict keys: ['mu'] ['logvar'] ['logits']

    kl_gauss_comps = kl_gauss_diag(
        var_params, prior_params
    )  # rets B*K; or shape*K (if input shape*K*D)

    kl_categ_comps = kl_categ(
        var_params["logits"], prior_params["logits"]
    )  # rets B*1; or shape*1

    mix_var_probs = F.softmax(
        var_params["logits"], dim=-1
    )  # B*K; or shape*K (categ. dist, controls comp. assignment)

    weighted_kl = (mix_var_probs * kl_gauss_comps).sum(
        dim=-1, keepdim=True
    ) + kl_categ_comps  # .view(-1,1) or .view(shape, 1)

    return weighted_kl, kl_categ_comps, kl_gauss_comps  # rets Bx1;   otherwise .sum()


def kl_empirical():

    # TODO: log q - log p -- like KL div loss;

    pass


def nll_batch_noreduce(dataset_obj, input_data, recon_params):

    if dataset_obj.dataset_type == "image":
        # image datasets, large number of features (vectorize loss)
        # compute NLL
        if dataset_obj.use_binary_img:
            nll_val = nll_binary_global(logit_fn(recon_params["x"]), input_data).sum(
                dim=-1
            )

        else:
            nll_val = nll_gauss_global(
                recon_params["x"],
                input_data,
                recon_params["logvar_x"],
                shape_feats=[len(dataset_obj.num_cols)],
            ).sum(dim=-1)

        nll_val = nll_val.view(-1, 1)  # Bx1

    else:
        start = 0
        cursor_num_feat = 0
        nll_val = torch.zeros(input_data.shape[0], 1).type(input_data.type())  # Bx1

        for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):
            # compute NLL
            if col_type == "categ":
                nll_val += nll_categ_global(
                    recon_params["x"][:, start : (start + feat_size)],
                    input_data[:, feat_select].long(),
                )
                start += feat_size

            elif col_type == "real":
                nll_val += nll_gauss_global(
                    recon_params["x"][:, start : (start + 1)],  # 2
                    input_data[:, feat_select],
                    recon_params["logvar_x"][:, cursor_num_feat],
                    shape_feats=[1],
                )

                start += 1  # 2
                cursor_num_feat += 1

    return nll_val  # Bx1


def y_weighted_nll(dataset_obj, input_data, recon_p_y1, recon_p_y0, y_prob):

    # y_prob: Bx1

    _nll = y_prob.view(-1, 1) * nll_batch_noreduce(dataset_obj, input_data, recon_p_y1)

    _nll += (1.0 - y_prob.view(-1, 1)) * nll_batch_noreduce(
        dataset_obj, input_data, recon_p_y0
    )

    return _nll  # rets Bx1


##### Combined Losses for Models


def vae_standard_ELBO(dataset_obj, input_data, p_params, q_params, kl_coeff=1.0):

    kl_z = kl_gauss_diag(q_params["z"], p_params["z"]).sum()

    nll_x = nll_batch_noreduce(dataset_obj, input_data, p_params["recon"]).sum()

    loss = kl_coeff * kl_z + nll_x

    return loss, nll_x, kl_z  # scalar


def n_comp_vae_gmm_ELBO(
    dataset_obj,
    input_data,
    p_params,
    q_params,
    kl_coeff=1.0,
    n_comps=None,
    data_size=None,
):
    """ 
    -- ELBO loss for n-component VAE GMM, i.e. several Gaussian components all can be learnt.

    -- Inspired by general VAE-GMM formulation by Rui Shu et al.

    NOTE: expectation through enumeration for categorical variable 
    """

    if not data_size:
        data_size = q_params["z_a"]["mu"].shape[0]

    if not n_comps:
        n_comps = q_params["z_a"]["mu"].shape[1]

    _var_params = {
        "mu": q_params["z_a"]["mu"],  # BxKxD
        "logvar": q_params["z_a"]["logvar"],  # BxKxD
        "logits": q_params["a"]["logits"],  # BxK
    }

    _prior_params = {
        "mu": p_params["z"]["mu"].unsqueeze(0),
        "logvar": p_params["z"]["logvar"].unsqueeze(0),
        "logits": p_params["a"]["logits"],
    }

    kld_tot, kld_a, kld_z = kl_gauss_diag_mixture(_var_params, _prior_params)  # .sum()
    kld_gmm = kld_tot.sum()

    _nll_batch = nll_batch_noreduce(dataset_obj, input_data, p_params["recon_a"])
    _nll_batch = _nll_batch.view(data_size, n_comps)  # [B,K]

    _var_prob_a = F.softmax(q_params["a"]["logits"], dim=-1)  # [B,K]

    nll_x = (_var_prob_a * _nll_batch).sum()

    loss = kl_coeff * kld_gmm + nll_x

    return loss, nll_x, kld_gmm, kld_a.sum()


def semi_y_vae_ELBO(
    dataset_obj,
    input_data,
    p_params,
    q_params,
    y_targets=None,
    mask_semi=None,
    kl_coeff=1.0,
    isdiag_q_z_y1=True,
    isdiag_q_z_y0=True,
    isdiag_p_z_y1=True,
    isdiag_p_z_y0=True,
    q_samples=None,
):
    # Negative ELBO for: p(x) + p(x,y)

    _var_y_logit = q_params["y"]["logits"]
    _var_y_prob = torch.sigmoid(_var_y_logit)
    _prior_y_logit = p_params["y"]["logits"]

    _var_z_y1_params = q_params["z_y1"]
    _var_z_y0_params = q_params["z_y0"]
    _prior_z_y1_params = p_params["z_y1"]
    _prior_z_y0_params = p_params["z_y0"]

    _recon_p_y1 = p_params["recon_y1"]
    _recon_p_y0 = p_params["recon_y0"]

    # UNSUPERVISED ELBO
    kld_y = kl_bern(_var_y_logit, _prior_y_logit).view(-1, 1)

    if isdiag_q_z_y1:
        kld_z_y1 = kl_gauss_diag(_var_z_y1_params, _prior_z_y1_params).view(-1, 1)
    else:
        kld_z_y1 = kl_gauss_full_cov(
            _var_z_y1_params, _prior_z_y1_params, is_prior_diag=isdiag_p_z_y1
        ).view(-1, 1)

    if isdiag_q_z_y0:
        kld_z_y0 = kl_gauss_diag(_var_z_y0_params, _prior_z_y0_params).view(-1, 1)
    else:
        kld_z_y0 = kl_gauss_full_cov(
            _var_z_y0_params, _prior_z_y0_params, is_prior_diag=isdiag_p_z_y0
        ).view(-1, 1)

    unsup_kld_z_y = _var_y_prob * kld_z_y1 + (1.0 - _var_y_prob) * kld_z_y0

    unsup_nll_x = y_weighted_nll(
        dataset_obj, input_data, _recon_p_y1, _recon_p_y0, _var_y_prob
    ).view(-1, 1)

    unsup_kld_tot = unsup_kld_z_y + kld_y
    unsup_loss = unsup_nll_x + kl_coeff * unsup_kld_tot

    # SUPERVISED ELBO
    if (mask_semi is None) or (y_targets is None):
        _mask_semi = 0.0

        nlog_p_y = 0.0
        sup_kld_z_y = 0.0
        sup_nll_x = 0.0
        sup_kld_tot = 0.0
        sup_loss = 0.0
    else:
        _y_targets = y_targets.view(-1, 1).float()
        _mask_semi = mask_semi.view(-1, 1).float()

        nlog_p_y = -torch.log(_var_y_prob + EPS).view(-1, 1)

        sup_kld_z_y = _y_targets * kld_z_y1 + (1.0 - _y_targets) * kld_z_y0

        sup_nll_x = y_weighted_nll(
            dataset_obj, input_data, _recon_p_y1, _recon_p_y0, _y_targets
        ).view(-1, 1)

        sup_kld_tot = sup_kld_z_y + nlog_p_y
        sup_loss = sup_nll_x + kl_coeff * sup_kld_tot

    # COMBINE ELBO LOSSES (choose between unsup and sup given trusted mask)
    comb_loss = ((1.0 - _mask_semi) * unsup_loss + _mask_semi * sup_loss).sum()
    comb_nll_x = ((1.0 - _mask_semi) * unsup_nll_x + _mask_semi * sup_nll_x).sum()
    comb_kld_tot = ((1.0 - _mask_semi) * unsup_kld_tot + _mask_semi * sup_kld_tot).sum()
    comb_kld_y = ((1.0 - _mask_semi) * kld_y + _mask_semi * nlog_p_y).sum()

    return (
        comb_loss,
        comb_nll_x,
        comb_kld_tot,
        comb_kld_y,
        kld_z_y1.sum(),
        kld_z_y0.sum(),
    )


def semi_y_vae_partitioned_ELBO(
    dataset_obj,
    input_data,
    p_params,
    q_params,
    y_targets=None,
    mask_semi=None,
    kl_coeff=1.0,
    q_samples=None,
    dist_corr_reg=False,
    dist_corr_reg_coeff=1.0,
    reg_scheduler_val=1.0,
    n_epoch=None,
):

    # Negative ELBO for: p(x) + p(x,y)

    _var_y_logit = q_params["y"]["logits"]
    _var_y_prob = torch.sigmoid(_var_y_logit)
    _prior_y_logit = p_params["y"]["logits"]

    _recon_p_y1 = p_params["recon_y1"]
    _recon_p_y0 = p_params["recon_y0"]

    _var_z_c_params = q_params["z_clean"]  # "z_c"
    _prior_z_c_params = p_params["z_clean"]

    _var_z_d_params = q_params["z_dirty"]  # "z_d"
    _prior_z_d_params = p_params["z_dirty"]

    # UNSUPERVISED ELBO
    kld_y = kl_bern(_var_y_logit, _prior_y_logit).view(-1, 1)

    kld_z_c = kl_gauss_diag(_var_z_c_params, _prior_z_c_params).view(-1, 1)

    kld_z_d = kl_gauss_diag(_var_z_d_params, _prior_z_d_params).view(-1, 1)

    unsup_nll_x = y_weighted_nll(
        dataset_obj, input_data, _recon_p_y1, _recon_p_y0, _var_y_prob
    ).view(-1, 1)

    unsup_kld_tot = kld_z_d + kld_z_c + kld_y
    unsup_loss = unsup_nll_x + kl_coeff * unsup_kld_tot

    # SUPERVISED ELBO
    if (mask_semi is None) or (y_targets is None):
        _mask_semi = 0.0
        nlog_p_y = 0.0
        sup_nll_x = 0.0
        sup_kld_tot = 0.0
        sup_loss = 0.0

    else:
        _y_targets = y_targets.view(-1, 1).float()
        _mask_semi = mask_semi.view(-1, 1).float()

        nlog_p_y = -torch.log(_var_y_prob + EPS).view(-1, 1)

        sup_nll_x = y_weighted_nll(
            dataset_obj, input_data, _recon_p_y1, _recon_p_y0, _y_targets
        ).view(-1, 1)

        sup_kld_tot = kld_z_d + kld_z_c + nlog_p_y
        sup_loss = sup_nll_x + kl_coeff * sup_kld_tot

    # COMBINE ELBO LOSSES (choose between unsup and sup given trusted mask)
    comb_loss = ((1.0 - _mask_semi) * unsup_loss + _mask_semi * sup_loss).sum()
    comb_nll_x = ((1.0 - _mask_semi) * unsup_nll_x + _mask_semi * sup_nll_x).sum()
    comb_kld_tot = ((1.0 - _mask_semi) * unsup_kld_tot + _mask_semi * sup_kld_tot).sum()
    comb_kld_y = ((1.0 - _mask_semi) * kld_y + _mask_semi * nlog_p_y).sum()

    _len_batch = q_samples["z_dirty"].shape[0]

    # ADD DISTANCE CORRELATION REGULARIZATION
    _dist_corr = distance_correlation(
        q_samples["z_clean"], q_samples["z_dirty"], squared=False
    )

    if dist_corr_reg:
        comb_loss = (
            comb_loss
            + reg_scheduler_val * dist_corr_reg_coeff * _len_batch * _dist_corr
        )

    # additional info
    with torch.no_grad():
        kld_z_y1 = kld_z_c.sum()
        kld_z_y0 = kld_z_c.sum() + kld_z_d.sum()

    return (
        comb_loss,
        comb_nll_x,
        comb_kld_tot,
        comb_kld_y,
        kld_z_y1,
        kld_z_y0,
        _dist_corr * _len_batch,
    )


##### IWAE Loss (Approximate of log_px -- log prob marginal of data)


def log_px_approx_eval(
    model,
    dataset_obj,
    x_inputs,
    k_samples=100,
    y_comps=False,
    batch_size=128,
    y_targets=None,
    repair_mode=True,
):

    if y_targets is not None:
        dataset_in = torch.utils.data.TensorDataset(x_inputs, y_targets)
    else:
        dataset_in = torch.utils.data.TensorDataset(x_inputs)

    dt_input_loader = torch.utils.data.DataLoader(
        dataset_in, batch_size=batch_size, shuffle=False
    )

    model.train()

    iwae_loss_list = []

    with torch.no_grad():
        for data_batch in tqdm(dt_input_loader, desc="Compute Progress", ncols=100):

            _data_in = data_batch[0]

            if y_targets is not None:
                _y_targets_batch = data_batch[1]
            else:
                _y_targets_batch = None

            if y_comps:
                # gets dict() with iwae of both y components
                elbo_iwae_map = log_px_iwae_y_comps(
                    model, dataset_obj, _data_in, k_samples=k_samples
                )

                elbo_iwae = (elbo_iwae_map["y1"], elbo_iwae_map["y0"])
                iwae_loss_list.append(elbo_iwae)

            else:
                # gets standard iwae (or just y=1, clean component)
                elbo_iwae = log_px_iwae_basic(
                    model,
                    dataset_obj,
                    _data_in,
                    k_samples=k_samples,
                    y_targets=_y_targets_batch,
                    repair_mode=repair_mode,
                )
                iwae_loss_list.append(elbo_iwae)

    if y_comps:
        _iwae_y1, _iwae_y0 = list(zip(*iwae_loss_list))
        elbo_iwae_ret = {
            "y1": torch.cat(_iwae_y1, dim=0),
            "y0": torch.cat(_iwae_y0, dim=0),
        }

    else:
        elbo_iwae_ret = torch.cat(iwae_loss_list, dim=0)

    model.eval()

    return elbo_iwae_ret


def log_px_iwae_basic(
    model,
    dataset_obj,
    x_inputs,
    k_samples=100,
    y_targets=None,
    repair_mode=False,
):

    """ Uses IWAE loss to approximate log p(x) """

    n_data = x_inputs.shape[0]

    _x_inputs = repeat_rows(x_inputs, k_samples)

    if y_targets is not None:
        _y_targets = repeat_rows(y_targets.view(-1, 1), k_samples)
    else:
        _y_targets = None

    p_params, q_params, q_samples, _ = model(
        _x_inputs,
        y_targets=_y_targets,
        repair_mode=repair_mode,
    )

    _p_x = p_params["recon"]  # "x" & "logvar_x"
    _p_z = p_params["z"]

    _q_z = q_params["z"]
    _q_z_sps = q_samples["z"]

    # get distribution objects
    mdl_dists = model.get_z_dists()

    # get elbo (or w_k's of iwae loss)
    log_q_z = (
        mdl_dists["q_z"].log_probs(_q_z_sps, force_params=_q_z).view(-1, 1)
    )  # (BxK)x1
    log_p_z = (
        mdl_dists["p_z"].log_probs(_q_z_sps, force_params=_p_z).view(-1, 1)
    )  # (BxK)x1
    log_p_x = -1 * nll_batch_noreduce(dataset_obj, _x_inputs, _p_x)  # (BxK)x1

    elbo = log_p_x + log_p_z - log_q_z  # (BxK)x1
    elbo = split_leading_dim(elbo, (n_data, k_samples))  # splits to BxKx1

    elbo_iwae = log_mean_exp(elbo, 1).squeeze(1)  # Bx1 (collapse K)
    # NOTE: can do average (mean) of B outside of func, for dataset or batch

    return elbo_iwae


def log_px_iwae_y_comps(
    model,
    dataset_obj,
    x_inputs,
    k_samples=100,
):
    """ Uses IWAE loss to approximate log p(x) """

    n_data = x_inputs.shape[0]
    _x_inputs = repeat_rows(x_inputs, k_samples)

    p_params, q_params, q_samples, _ = model(_x_inputs)

    # get distribution objects
    _dists = model.get_z_dists()

    # get elbo (or w_k's of iwae loss)
    elbo_iwae_map = dict()
    for y_str in ["y1", "y0"]:
        _p_x = p_params["recon_" + y_str]

        if "p_z_c" in _dists:  # check key in mdl dict

            _q_z_c = q_params["z_clean"]
            _q_z_c_sps = q_samples["z_clean"]
            _p_z_c = p_params["z_clean"]

            _q_z_d = q_params["z_dirty"]
            _q_z_d_sps = q_samples["z_dirty"]
            _p_z_d = p_params["z_dirty"]

            log_q_z_c = (
                _dists["q_z_c"].log_probs(_q_z_c_sps, force_params=_q_z_c).view(-1, 1)
            )
            log_p_z_c = (
                _dists["p_z_c"].log_probs(_q_z_c_sps, force_params=_p_z_c).view(-1, 1)
            )

            log_p_x = -1 * nll_batch_noreduce(dataset_obj, _x_inputs, _p_x)  # (BxK)x1

            if y_str == "y0":
                log_q_z_d = (
                    _dists["q_z_d"]
                    .log_probs(_q_z_d_sps, force_params=_q_z_d)
                    .view(-1, 1)
                )
                log_p_z_d = (
                    _dists["p_z_d"]
                    .log_probs(_q_z_d_sps, force_params=_p_z_d)
                    .view(-1, 1)
                )

                elbo = (
                    log_p_x + log_p_z_c - log_q_z_c + log_p_z_d - log_q_z_d
                )  # (BxK)x1
                elbo = split_leading_dim(elbo, (n_data, k_samples))  # splits to BxKx1

            else:
                elbo = log_p_x + log_p_z_c - log_q_z_c  # (BxK)x1
                elbo = split_leading_dim(elbo, (n_data, k_samples))  # splits to BxKx1

        else:
            _p_z = p_params["z_" + y_str]
            _q_z = q_params["z_" + y_str]
            _q_z_sps = q_samples["z_" + y_str]

            if isinstance(_dists["q_z_" + y_str], GaussFullCovDistModule):
                _q_z["covar_chol"] = torch.cholesky(_q_z["covar"], upper=False)

            log_q_z = (
                _dists["q_z_" + y_str]
                .log_probs(_q_z_sps, force_params=_q_z)
                .view(-1, 1)
            )  # (BxK)x1

            log_p_z = (
                _dists["p_z_" + y_str]
                .log_probs(_q_z_sps, force_params=_p_z)
                .view(-1, 1)
            )  # (BxK)x1

            log_p_x = -1 * nll_batch_noreduce(dataset_obj, _x_inputs, _p_x)  # (BxK)x1

            elbo = log_p_x + log_p_z - log_q_z  # (BxK)x1
            elbo = split_leading_dim(elbo, (n_data, k_samples))  # splits to BxKx1

        elbo_iwae_map[y_str] = log_mean_exp(elbo, 1).squeeze(1)  # Bx1 (collapse K)
        # NOTE: can do average (mean) of B outside of func, for dataset or batch

    return elbo_iwae_map


###########

# Distance Correlation

def distance_correlation(x_sps, y_sps, squared=False):

    n_samples = x_sps.shape[0]

    a = torch.norm(x_sps.unsqueeze(-2) - x_sps, dim=-1, p=2)
    b = torch.norm(y_sps.unsqueeze(-2) - y_sps, dim=-1, p=2)

    A = a - a.mean(dim=0).unsqueeze(0) - a.mean(dim=1).unsqueeze(1) + a.mean()
    B = b - b.mean(dim=0).unsqueeze(0) - b.mean(dim=1).unsqueeze(1) + b.mean()

    dcov2_xx = (A * A).sum() / float(n_samples ** 2)
    dcov2_yy = (B * B).sum() / float(n_samples ** 2)
    dcov2_xy = (A * B).sum() / float(n_samples ** 2)

    if squared:
        dist_corr = dcov2_xy / (torch.sqrt(dcov2_yy) * torch.sqrt(dcov2_xx))

    else:
        dist_corr = torch.sqrt(dcov2_xy) / torch.sqrt(
            torch.sqrt(dcov2_yy) * torch.sqrt(dcov2_xx)
        )

    return dist_corr


# MMD Loss

class MMD_loss(nn.Module):
    """ Square MMD loss """

    def __init__(self):
        super().__init__()

    def guassian_kernel(self, source, target):

        # n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)

        with torch.no_grad():
            _median_l2 = L2_distance.median()
            _log_n = torch.log(
                torch.tensor(source.shape[0], device=source.device) + EPS
            )  # + 1 ?

            # heuristic for bandwidth: see https://arxiv.org/abs/1608.04471
            bandwidth = torch.sqrt(0.5 * _median_l2 / _log_n)

        kernel_val = torch.exp(-L2_distance / (2 * (bandwidth ** 2)))

        return kernel_val, bandwidth

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels, bandwidth = self.guassian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss, bandwidth
