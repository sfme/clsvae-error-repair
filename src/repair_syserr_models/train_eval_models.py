
import torch
from collections import OrderedDict
import repair_syserr_models.gen_utils as gen_utils

EPS = 1e-6


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def training_phase(
    model,
    trainer_optim,
    train_loader,
    args,
    epoch,
    mask_semi=None,
    kl_beta=1.0,
    reg_schedule_val=1.0,
):

    model.train()

    metrics_dict = None

    for batch_idx, unpack in enumerate(train_loader):

        data_input = unpack[0]
        # NOTE:
        # unpack[1] -> error mask of batch (which cells / pixels are corrupted)
        # unpack[2] -> indices of current batch related to dataset

        if args.cuda_on:
            data_input = data_input.cuda()

        # if semi-supervised training (get labelled instances mask for train)
        if args.semi_supervise:
            idxs_batch = unpack[2]
            mask_semi_batch = mask_semi[idxs_batch].view(-1, 1)
        else:
            mask_semi_batch = None

        err_cell_mask = unpack[1].bool().to(data_input.device) # ground-truth of cell / pixel errors 
        err_row_mask = err_cell_mask.any(dim=-1).view(-1, 1)
        _targets_sup = ~err_row_mask # ground-truth labels (inlier / outlier)

        # TODO: make semi_supervise flag method dependent?
        loss_dict = trainer_optim.train_step(
            data_input,
            _targets_sup,
            epoch,
            kl_beta,
            mask_semi_batch,
            reg_schedule_val,
        )

        if metrics_dict is None:
            metrics_dict = dict((_key, 0.0) for _key in loss_dict.keys())

        metrics_dict = {
            _key: (_value + loss_dict[_key].item())
            for _key, _value in metrics_dict.items()
        }

        if batch_idx % args.log_interval == 0:

            out_string = "\n\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tELBO (or Total) Loss: {:.3f}\t".format(
                epoch,
                batch_idx * len(data_input),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss_dict["total_loss"].item() / len(data_input),
            )

            if args.semi_supervise:
                _loss_sup = loss_dict["loss_sup"].item() / (mask_semi_batch.sum() + EPS)

                out_string += "{}: {:.3f}\t".format(
                    "loss_sup".upper(),
                    _loss_sup,
                )

            out_string += "".join(
                [
                    "{}: {:.3f}\t".format(_key.upper(), _value.item() / len(data_input))
                    for _key, _value in loss_dict.items()
                    if _key not in ["total_loss", "loss_sup"]
                ]
            )

            print(out_string)

    dataset_len = float(len(train_loader.dataset))

    ret_list = [
        ("train_" + _key, _value / dataset_len)
        for _key, _value in metrics_dict.items()
        if _key != "loss_sup"
    ]

    if args.semi_supervise:
        # size of labelled set (number instances)
        sup_dataset_len = mask_semi.sum().item()
        # cross-entropy loss (average)
        ret_list.insert(
            2, ("train_" + "loss_sup", metrics_dict["loss_sup"] / sup_dataset_len)
        )

    ret = OrderedDict(ret_list)

    return ret


def evaluation_phase(
    model,
    data_loader,
    data_eval,
    dataset_obj,
    args,
    mode,
    mask_semi=None,
    kl_beta=1.0,
    reg_schedule_val=1.0,
    mask_err=None,
):

    model.eval()

    with torch.no_grad():
        _recons_x_list = []
        _q_y_logits_list = []
        loss_dict = None

        for batch_idx, unpack in enumerate(data_loader):

            data_input = unpack[0]
            # unpack[1] # error mask of batch (which cells / pixels are corrupted)
            # unpack[2] # indices of current batch related to dataset

            if args.cuda_on:
                data_input = data_input.cuda()

            # if semi-supervised training
            if args.semi_supervise and (mode == "train"):
                idxs_batch = unpack[2]
                mask_semi_batch = mask_semi[idxs_batch].view(-1, 1)

            else:
                mask_semi_batch = None

            err_cell_mask = unpack[1].bool().to(data_input.device)
            err_row_mask = err_cell_mask.any(dim=-1).view(-1, 1)
            _targets_sup = ~err_row_mask

            p_params, q_params, q_samples, log_q_dists = model(
                data_input, y_targets=_targets_sup
            )

            _loss_dict = model.loss_function(
                data_input,
                p_params,
                q_params,
                q_samples,
                log_q_dists,
                mask_semi_batch,
                _targets_sup,
                sup_coeff=args.sup_loss_coeff,
                kl_coeff=kl_beta,
                reg_scheduler_val=reg_schedule_val,
            )

            if loss_dict is None:
                loss_dict = dict((_key, 0.0) for _key in _loss_dict.keys())

            loss_dict = {
                _key: (_value + _loss_dict[_key].item())
                for _key, _value in loss_dict.items()
            }

            _recons_x_list.append(p_params["recon"]["x"])
            if args.semi_supervise:
                _q_y_logits_list.append(q_params["y"]["logits"])

        _recons = dict()
        _recons["x"] = torch.cat(_recons_x_list, dim=0)

        if ("logvar_x" in p_params["recon"]) and isinstance(
            p_params["recon"]["logvar_x"], torch.Tensor
        ):
            _recons["logvar_x"] = p_params["recon"]["logvar_x"]

        if args.semi_supervise:
            _q_y_logits = torch.cat(_q_y_logits_list, dim=0)

    eval_datalen = len(data_loader.dataset)

    losses_list = [
        (_key, _value / eval_datalen)
        for _key, _value in loss_dict.items()
        if _key != "loss_sup"
    ]

    if args.semi_supervise and (mode == "train"):
        sup_dataset_len = mask_semi.sum().item()
        losses_list.insert(2, ("loss_sup", loss_dict["loss_sup"] / sup_dataset_len))
    else:
        losses_list.insert(2, ("loss_sup", 0.0))
        # NOTE: if not semi-sup; or test set (no trusted set there)

    losses = OrderedDict(losses_list)

    # using decoder (or likelihood) for outlier detection
    nll_score_mat = gen_utils.generate_score_outlier_matrix(
        _recons, data_eval, dataset_obj
    )

    # TODO: make decision about using q_y prediction based on model name, instead of semi_supervise flag?
    if args.semi_supervise:
        class_y_score = gen_utils.gen_score_OD_class_y(_q_y_logits)
    else:
        class_y_score = -10.0

    metrics = OrderedDict(
        [("nll_score", nll_score_mat), ("class_y_score", class_y_score)]
    )

    # losses and metrics specific to trusted set (for evaluating labelled set performance)
    if (mask_semi is not None) and (mode == "train"):

        with torch.no_grad():
            _data_sup = data_eval[mask_semi]
            _eval_datalen_sup = _data_sup.shape[0]

            _chunks_idxs = chunks(list(range(_eval_datalen_sup)), args.batch_size)

            if args.semi_supervise:
                _mask_all_ones = torch.ones((_eval_datalen_sup, 1)).to(data_eval.device)
            else:
                _mask_all_ones = None

            _y_targets_sup = ~(mask_err[mask_semi].bool().any(dim=-1).view(-1, 1))

            loss_dict_sup_acc = None

            for batch_idxs in _chunks_idxs:
                if _y_targets_sup is not None:
                    _yts = _y_targets_sup[batch_idxs]
                else:
                    _yts = None

                if _mask_all_ones is not None:
                    _ones = _mask_all_ones[batch_idxs]
                else:
                    _ones = None

                _p_params_sup, _q_params_sup, _q_samples_sup, _log_q_dists_sup = model(
                    _data_sup[batch_idxs], y_targets=_yts
                )

                _loss_dict_sup = model.loss_function(
                    _data_sup[batch_idxs],
                    _p_params_sup,
                    _q_params_sup,
                    _q_samples_sup,
                    _log_q_dists_sup,
                    _ones,
                    _yts,
                    sup_coeff=args.sup_loss_coeff,
                    kl_coeff=kl_beta,
                    reg_scheduler_val=reg_schedule_val,
                )

                if loss_dict_sup_acc is None:
                    loss_dict_sup_acc = dict(
                        (_key, 0.0) for _key in _loss_dict_sup.keys()
                    )

                loss_dict_sup_acc = {
                    _key: (_value + _loss_dict_sup[_key].item())
                    for _key, _value in loss_dict_sup_acc.items()
                }

        losses_list_sup = [
            (_key, _value / _eval_datalen_sup)
            for _key, _value in loss_dict_sup_acc.items()
        ]

        if not args.semi_supervise:
            losses_list_sup.insert(2, ("loss_sup", 0.0))

        losses_trusted_set = OrderedDict(losses_list_sup)

        if args.semi_supervise:
            class_y_score_ts = class_y_score[mask_semi]
        else:
            class_y_score_ts = -10.0

        metrics_trusted_set = OrderedDict(
            [
                ("nll_score", nll_score_mat[mask_semi]),
                ("class_y_score", class_y_score_ts),
            ]
        )

    else:
        losses_trusted_set = None
        metrics_trusted_set = None

    return losses, metrics, losses_trusted_set, metrics_trusted_set


def repair_phase(
    model, data_dirty, data_clean, dataset_obj, args, mask_err, mode, mask_semi=None
):

    model.eval()

    outlier_dts = mask_err.any(dim=1)
    y_targets = ~outlier_dts.view(-1, 1)

    with torch.no_grad():
        # model params with input: dirty data
        p_params_xd, q_params_xd, q_samples_xd, _ = model(
            data_dirty, y_targets=y_targets, repair_mode=True
        )

        # model params with input: underlying clean data
        p_params_xc, q_params_xc, q_samples_xc, _ = model(
            data_clean, y_targets=y_targets, repair_mode=True
        )

    # error (MSE) lower bound, on dirty cell positions only
    error_lb_dc, error_lb_dc_per_feat = gen_utils.error_computation(
        model, data_clean, p_params_xc["recon"]["x"], mask_err
    )  # x_truth - f_vae(x_clean)

    # error repair, on dirty cell positions only
    error_repair_dc, error_repair_dc_per_feat = gen_utils.error_computation(
        model, data_clean, p_params_xd["recon"]["x"], mask_err
    )  # x_truth - f_vae(x_dirty)

    print("\n\n {} REPAIR ERROR (DIRTY POS):{}".format(mode, error_repair_dc))

    # error upper bound, on dirty cell positions only
    error_up_dc, error_up_dc_per_feat = gen_utils.error_computation(
        model, data_clean, data_dirty, mask_err, x_input_resize=True
    )  # x_truth - x_dirty

    # error on clean cell positions only (to test impact of dirty cells on clean cells under model)
    error_repair_cc, error_repair_cc_per_feat = gen_utils.error_computation(
        model, data_clean, p_params_xd["recon"]["x"], ~mask_err
    )

    print("\n\n {} REPAIR ERROR (CLEAN POS):{}".format(mode, error_repair_cc))

    # error on clean cell positions on dirty datapoints (to measure distortion in repair)
    error_repair_cc_out, error_repair_cc_per_feat_out = gen_utils.error_computation(
        model,
        data_clean[outlier_dts, :],
        p_params_xd["recon"]["x"][outlier_dts, :],
        ~mask_err[outlier_dts, :],
    )

    # trusted set repairs (for analysis)
    if (mask_semi is not None) and (mode == "train"):

        with torch.no_grad():
            # model params with input: dirty data
            p_params_xd_ts, q_params_xd_ts, q_samples_xd_ts, _ = model(
                data_dirty[mask_semi, :],
                y_targets=y_targets[mask_semi, :],
                repair_mode=True,
            )

            # model params with input: underlying clean data
            p_params_xc_ts, q_params_xc_ts, q_samples_xc_ts, _ = model(
                data_clean[mask_semi, :],
                y_targets=y_targets[mask_semi, :],
                repair_mode=True,
            )

        # error (MSE) lower bound, on dirty cell positions only
        error_lb_dc_ts, error_lb_dc_per_feat_ts = gen_utils.error_computation(
            model,
            data_clean[mask_semi, :],
            p_params_xc_ts["recon"]["x"],
            mask_err[mask_semi, :],
        )  # x_truth - f_vae(x_clean)

        # error repair, on dirty cell positions only
        error_repair_dc_ts, error_repair_dc_per_feat_ts = gen_utils.error_computation(
            model,
            data_clean[mask_semi, :],
            p_params_xd_ts["recon"]["x"],
            mask_err[mask_semi, :],
        )  # x_truth - f_vae(x_dirty)

        print(
            "\n\n {} (trusted set) REPAIR ERROR (DIRTY POS):{}".format(
                mode, error_repair_dc_ts
            )
        )

        # error upper bound, on dirty cell positions only
        error_up_dc_ts, error_up_dc_per_feat_ts = gen_utils.error_computation(
            model,
            data_clean[mask_semi, :],
            data_dirty[mask_semi, :],
            mask_err[mask_semi, :],
            x_input_resize=True,
        )  # x_truth - x_dirty

        # error on clean cell positions only (to test impact of dirty cells on clean cells under model)
        error_repair_cc_ts, error_repair_cc_per_feat_ts = gen_utils.error_computation(
            model,
            data_clean[mask_semi, :],
            p_params_xd_ts["recon"]["x"],
            ~mask_err[mask_semi, :],
        )

        # error on clean cell positions on dirty datapoints (to measure distortion in repair)
        _select_idxs_ts_out = outlier_dts.view(-1, 1) & mask_semi.view(-1, 1)
        _select_idxs_ts_out = _select_idxs_ts_out.flatten()
        (
            error_repair_cc_out_ts,
            error_repair_cc_per_feat_out_ts,
        ) = gen_utils.error_computation(
            model,
            data_clean[_select_idxs_ts_out, :],
            p_params_xd_ts["recon"]["x"][outlier_dts[mask_semi], :],
            ~mask_err[_select_idxs_ts_out, :],
        )

        print(
            "\n\n {} (trusted set) REPAIR ERROR (CLEAN POS):{}".format(
                mode, error_repair_cc_ts
            )
        )

        losses_trusted_set = {
            "mse_lower_bd_dirtycells": error_lb_dc_ts.item(),
            "mse_upper_bd_dirtycells": error_up_dc_ts.item(),
            "mse_repair_dirtycells": error_repair_dc_ts.item(),
            "mse_repair_cleancells": error_repair_cc_ts.item(),
            "mse_repair_cleancells_outliers": error_repair_cc_out_ts.item(),
            "errors_per_feature": [
                error_lb_dc_per_feat_ts,
                error_repair_dc_per_feat_ts,
                error_up_dc_per_feat_ts,
                error_repair_cc_per_feat_ts,
                error_repair_cc_per_feat_out_ts,
            ],
        }

    else:
        losses_trusted_set = None

    losses = {
        "mse_lower_bd_dirtycells": error_lb_dc.item(),
        "mse_upper_bd_dirtycells": error_up_dc.item(),
        "mse_repair_dirtycells": error_repair_dc.item(),
        "mse_repair_cleancells": error_repair_cc.item(),
        "mse_repair_cleancells_outliers": error_repair_cc_out.item(),
        "errors_per_feature": [
            error_lb_dc_per_feat,
            error_repair_dc_per_feat,
            error_up_dc_per_feat,
            error_repair_cc_per_feat,
            error_repair_cc_per_feat_out,
        ],
    }

    return losses, losses_trusted_set
