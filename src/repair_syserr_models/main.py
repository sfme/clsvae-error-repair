import sys

import json
import os, errno
import pandas as pd
import numpy as np

import torch

import repair_syserr_models.gen_utils as gen_utils

import repair_syserr_models.parser_arguments as parser_arguments
from repair_syserr_models.train_eval_models import training_phase, evaluation_phase, repair_phase

from repair_syserr_models.model_utils import frange_cycle_linear

from repair_syserr_models.trainer_utils import StandardTrainer


def compute_metrics(
    model,
    data_loader_X,
    X,
    dataset_obj,
    args,
    epoch,
    losses_save,
    X_clean,
    target_errors,
    trusted_mask,
    mode,
    kl_beta=1.0,
    reg_scheduler_val=1.0,
):

    # get epoch metrics on outlier detection for train dataset

    # outlier analysis
    loss_ret, metric_ret, loss_trusted_ret, metric_trusted_ret = evaluation_phase(
        model,
        data_loader_X,
        X,
        dataset_obj,
        args,
        mode,
        trusted_mask,
        kl_beta,
        reg_scheduler_val,
        target_errors,
    )

    # repair analysis
    repair_ret, repair_trusted_ret = repair_phase(
        model,
        X,
        X_clean,
        dataset_obj,
        args,
        target_errors,
        mode,
        trusted_mask,
    )

    print("\n\n\n\n")
    out_string = (
        "====> "
        + mode
        + " set: Epoch: {} Avg. Loss: {:.3f}\t".format(epoch, loss_ret["total_loss"])
    )
    out_string += "".join(
        [
            "{}: {:.3f}\t".format("Avg. " + _key.upper(), _value)
            for _key, _value in loss_ret.items()
            if _key != "total_loss"
        ]
    )
    print(out_string)

    # calc cell metrics
    auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = gen_utils.cell_metrics(
        target_errors, metric_ret["nll_score"]
    )

    # calc row metrics
    auc_row_nll, avpr_row_nll = gen_utils.row_metrics(
        target_errors, metric_ret["nll_score"]
    )

    # trusted set analysis
    if trusted_mask is not None:
        # calc cell metrics
        (
            auc_cell_nll_ts,
            auc_vec_nll_ts,
            avpr_cell_nll_ts,
            avpr_vec_nll_ts,
        ) = gen_utils.cell_metrics(
            target_errors[trusted_mask, :], metric_trusted_ret["nll_score"]
        )

        # calc row metrics
        auc_row_nll_ts, avpr_row_nll_ts = gen_utils.row_metrics(
            target_errors[trusted_mask], metric_trusted_ret["nll_score"]
        )

    else:
        auc_cell_nll_ts, auc_vec_nll_ts, avpr_cell_nll_ts, avpr_vec_nll_ts = (
            -10.0,
            -10.0,
            -10.0,
            -10.0,
        )
        auc_row_nll_ts, avpr_row_nll_ts = -10.0, -10.0

    if args.semi_supervise:  # "semi_y_VAE" in args.model_type
        # TODO: or any other model with classifier like score.
        auc_row_class_y, avpr_row_class_y = gen_utils.row_metrics_classifier(
            target_errors, metric_ret["class_y_score"]
        )
        if trusted_mask is not None:
            auc_row_class_y_ts, avpr_row_class_y_ts = gen_utils.row_metrics_classifier(
                target_errors, metric_ret["class_y_score"]
            )
        else:
            auc_row_class_y_ts, avpr_row_class_y_ts = -10.0, -10.0
    else:
        auc_row_class_y = -10.0
        avpr_row_class_y = -10.0
        auc_row_class_y_ts = -10.0
        avpr_row_class_y_ts = -10.0

    if args.verbose_metrics_epoch:
        print("         (Cell) Avg. " + mode + " AUC: {} ".format(auc_cell_nll))
        print("         (Cell) Avg. " + mode + " AVPR: {} ".format(avpr_cell_nll))
        print("\n\n")
        if args.verbose_metrics_feature_epoch:
            # TODO: might want to restrict if image?
            print("         AUC per feature: \n {}".format(auc_vec_nll))
            print("         AVPR per feature: \n {}".format(avpr_vec_nll))
            print("\n\n")
        print("         (Row) " + mode + " AUC: {} ".format(auc_row_nll))
        print("         (Row) " + mode + " AVPR: {} ".format(avpr_row_nll))
        print("\n\n")

        if args.semi_supervise:
            print(
                "         (Row) " + mode + " CLASSF_Y AUC: {} ".format(auc_row_class_y)
            )
            print(
                "         (Row) "
                + mode
                + " CLASSF_Y AVPR: {} ".format(avpr_row_class_y)
            )
            print("\n\n")
        print(
            "         (Cell) SMSE "
            + mode
            + " Lower Bound (on dirty pos): {:.3f}".format(
                repair_ret["mse_lower_bd_dirtycells"]
            )
        )
        print(
            "         (Cell) SMSE "
            + mode
            + " Upper Bound (on dirty pos): {:.3f}".format(
                repair_ret["mse_upper_bd_dirtycells"]
            )
        )
        print(
            "         (Cell) SMSE "
            + mode
            + " Repair (on dirty pos): {:.3f}".format(
                repair_ret["mse_repair_dirtycells"]
            )
        )
        print(
            "         (Cell) SMSE "
            + mode
            + " Repair (on clean pos): {:.3f}".format(
                repair_ret["mse_repair_cleancells"]
            )
        )
        print(
            "         (Cell) SMSE "
            + mode
            + " Repair (on clean pos for dirty points): {:.3f}".format(
                repair_ret["mse_repair_cleancells_outliers"]
            )
        )
        print("\n\n")

    if trusted_mask is not None:
        print("\n\n")
        out_string = "====> trusted set: Epoch: {} Avg. Loss: {:.3f}\t".format(
            epoch, loss_trusted_ret["total_loss"]
        )
        out_string += "".join(
            [
                "{}: {:.3f}\t".format("Avg. " + _key.upper(), _value)
                for _key, _value in loss_trusted_ret.items()
                if _key != "total_loss"
            ]
        )
        print(out_string)

        if args.verbose_metrics_epoch:
            print(
                "         (Cell) Avg. "
                + mode
                + " (trusted set) AUC: {} ".format(auc_cell_nll_ts)
            )
            print(
                "         (Cell) Avg. "
                + mode
                + " (trusted set) AVPR: {} ".format(avpr_cell_nll_ts)
            )
            print("\n\n")
            if args.verbose_metrics_feature_epoch:
                # TODO: might want to restrict if image dataset?
                print("         AUC per feature: \n {}".format(auc_vec_nll_ts))
                print("         AVPR per feature: \n {}".format(avpr_vec_nll_ts))
                print("\n\n")
            print(
                "         (Row) "
                + mode
                + " (trusted set) AUC: {} ".format(auc_row_nll_ts)
            )
            print(
                "         (Row) "
                + mode
                + " (trusted set) AVPR: {} ".format(avpr_row_nll_ts)
            )
            print("\n\n")
            if args.semi_supervise:
                print(
                    "         (Row) "
                    + mode
                    + " (trusted set) CLASSF_Y AUC: {} ".format(auc_row_class_y_ts)
                )
                print(
                    "         (Row) "
                    + mode
                    + " (trusted set) CLASSF_Y AVPR: {} ".format(avpr_row_class_y_ts)
                )
                print("\n\n")
            print(
                "         (Cell) SMSE "
                + mode
                + " (trusted set) Lower Bound (on dirty pos): {:.3f}".format(
                    repair_trusted_ret["mse_lower_bd_dirtycells"]
                )
            )
            print(
                "         (Cell) SMSE "
                + mode
                + " (trusted set) Upper Bound (on dirty pos): {:.3f}".format(
                    repair_trusted_ret["mse_upper_bd_dirtycells"]
                )
            )
            print(
                "         (Cell) SMSE "
                + mode
                + " (trusted set) Repair (on dirty pos): {:.3f}".format(
                    repair_trusted_ret["mse_repair_dirtycells"]
                )
            )
            print(
                "         (Cell) SMSE "
                + mode
                + " (trusted set) Repair (on clean pos): {:.3f}".format(
                    repair_trusted_ret["mse_repair_cleancells"]
                )
            )
            print(
                "         (Cell) SMSE "
                + mode
                + " (trusted set) Repair (on clean pos for dirty points): {:.3f}".format(
                    repair_trusted_ret["mse_repair_cleancells_outliers"]
                )
            )
            print("\n\n")

    if args.save_on:
        losses_save[mode][epoch] = list(loss_ret.values())
        losses_save[mode][epoch] += [
            auc_cell_nll,
            avpr_cell_nll,
            auc_row_nll,
            avpr_row_nll,
            auc_row_class_y,
            avpr_row_class_y,
            repair_ret["mse_lower_bd_dirtycells"],
            repair_ret["mse_upper_bd_dirtycells"],
            repair_ret["mse_repair_dirtycells"],
            repair_ret["mse_repair_cleancells"],
            repair_ret["mse_repair_cleancells_outliers"],
        ]
        if (mode == "train") and (trusted_mask is not None):
            losses_save["trusted"][epoch] = list(loss_trusted_ret.values())
            losses_save["trusted"][epoch] += [
                auc_cell_nll_ts,
                avpr_cell_nll_ts,
                auc_row_nll_ts,
                avpr_row_nll_ts,
                auc_row_class_y_ts,
                avpr_row_class_y_ts,
                repair_trusted_ret["mse_lower_bd_dirtycells"],
                repair_trusted_ret["mse_upper_bd_dirtycells"],
                repair_trusted_ret["mse_repair_dirtycells"],
                repair_trusted_ret["mse_repair_cleancells"],
                repair_trusted_ret["mse_repair_cleancells_outliers"],
            ]

        elif (mode == "train") and (trusted_mask is None):
            losses_save["trusted"][epoch] = [-10.0] * len(loss_ret.values())
            losses_save["trusted"][epoch] += [-10.0] * 11


def save_to_csv(
    model,
    data_loader_X,
    X_data,
    X_data_clean,
    target_errors,
    trusted_mask,
    attributes,
    losses_save,
    dataset_obj,
    folder_output,
    args,
    mode="train",
    kl_beta=1.0,
    reg_scheduler_val=1.0,
):

    """ This method performs all operations needed to save the data to csv """

    # Create saving folderes
    try:
        os.makedirs(folder_output, mode=0o777)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ### Evaluate model
    _, metric_ret, _, metric_trusted_ret = evaluation_phase(
        model,
        data_loader_X,
        X_data,
        dataset_obj,
        args,
        mode,
        trusted_mask,
        kl_beta,
        reg_scheduler_val,
        target_errors,
    )

    repair_ret, repair_trusted_ret = repair_phase(
        model,
        X_data,
        X_data_clean,
        dataset_obj,
        args,
        target_errors,
        mode,
        trusted_mask,
    )

    ## calc cell metrics
    auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = gen_utils.cell_metrics(
        target_errors, metric_ret["nll_score"]
    )

    # store AVPR for features (cell only)
    df_avpr_feat_cell = pd.DataFrame([], index=["AVPR_nll"], columns=attributes)
    df_avpr_feat_cell.loc["AVPR_nll"] = avpr_vec_nll
    df_avpr_feat_cell.to_csv(folder_output + "/" + mode + "_avpr_features.csv")

    # store AUC for features (cell only)
    df_auc_feat_cell = pd.DataFrame([], index=["AUC_nll"], columns=attributes)
    df_auc_feat_cell.loc["AUC_nll"] = auc_vec_nll
    df_auc_feat_cell.to_csv(folder_output + "/" + mode + "_auc_features.csv")

    # trusted set compute analysis
    if (trusted_mask is not None) and (mode == "train"):
        ## calc cell metrics
        (
            auc_cell_nll_ts,
            auc_vec_nll_ts,
            avpr_cell_nll_ts,
            avpr_vec_nll_ts,
        ) = gen_utils.cell_metrics(
            target_errors[trusted_mask, :], metric_trusted_ret["nll_score"]
        )

        # store AVPR for features (cell only)
        df_avpr_feat_cell_ts = pd.DataFrame([], index=["AVPR_nll"], columns=attributes)
        df_avpr_feat_cell_ts.loc["AVPR_nll"] = avpr_vec_nll_ts
        df_avpr_feat_cell_ts.to_csv(folder_output + "/trusted_avpr_features.csv")

        # store AUC for features (cell only)
        df_auc_feat_cell_ts = pd.DataFrame([], index=["AUC_nll"], columns=attributes)
        df_auc_feat_cell_ts.loc["AUC_nll"] = auc_vec_nll_ts
        df_auc_feat_cell_ts.to_csv(folder_output + "/trusted_auc_features.csv")

    ### Store data from Epochs
    columns = ["Avg. " + col_name.upper() for col_name in model.loss_ret_names]
    columns += [
        "AUC Cell nll score",
        "AVPR Cell nll score",
        "AUC Row nll score",
        "AVPR Row nll score",
        "AUC Row class_y score",
        "AVPR Row class_y score",
        "Error lower-bound on dirty pos",
        "Error upper-bound on dirty pos",
        "Error repair on dirty pos",
        "Error repair on clean pos",
        "Error repair on clean pos - dirty points",
    ]

    df_out = pd.DataFrame.from_dict(losses_save[mode], orient="index", columns=columns)
    df_out.index.name = "Epochs"
    df_out.to_csv(folder_output + "/" + mode + "_epochs_data.csv")

    if (trusted_mask is not None) and (mode == "train"):
        df_out_ts = pd.DataFrame.from_dict(
            losses_save["trusted"], orient="index", columns=columns
        )
        df_out_ts.index.name = "Epochs"
        df_out_ts.to_csv(folder_output + "/trusted_epochs_data.csv")

    ### Store errors per feature

    df_errors_repair = pd.DataFrame(
        [],
        index=[
            "error_lowerbound_dirtycells",
            "error_repair_dirtycells",
            "error_upperbound_dirtycells",
            "error_repair_cleancells",
            "error_repair_cleancells_dirtypoints",
        ],
        columns=attributes,
    )

    df_errors_repair.loc["error_lowerbound_dirtycells"] = repair_ret[
        "errors_per_feature"
    ][0].cpu()
    df_errors_repair.loc["error_repair_dirtycells"] = repair_ret["errors_per_feature"][
        1
    ].cpu()
    df_errors_repair.loc["error_upperbound_dirtycells"] = repair_ret[
        "errors_per_feature"
    ][2].cpu()
    df_errors_repair.loc["error_repair_cleancells"] = repair_ret["errors_per_feature"][
        3
    ].cpu()
    df_errors_repair.loc["error_repair_cleancells_dirtypoints"] = repair_ret[
        "errors_per_feature"
    ][4].cpu()

    df_errors_repair.to_csv(folder_output + "/" + mode + "_error_repair_features.csv")

    if (trusted_mask is not None) and (mode == "train"):

        df_errors_repair_ts = pd.DataFrame(
            [],
            index=[
                "error_lowerbound_dirtycells",
                "error_repair_dirtycells",
                "error_upperbound_dirtycells",
                "error_repair_cleancells",
                "error_repair_cleancells_dirtypoints",
            ],
            columns=attributes,
        )

        df_errors_repair_ts.loc["error_lowerbound_dirtycells"] = repair_trusted_ret[
            "errors_per_feature"
        ][0].cpu()
        df_errors_repair_ts.loc["error_repair_dirtycells"] = repair_trusted_ret[
            "errors_per_feature"
        ][1].cpu()
        df_errors_repair_ts.loc["error_upperbound_dirtycells"] = repair_trusted_ret[
            "errors_per_feature"
        ][2].cpu()
        df_errors_repair_ts.loc["error_repair_cleancells"] = repair_trusted_ret[
            "errors_per_feature"
        ][3].cpu()
        df_errors_repair_ts.loc[
            "error_repair_cleancells_dirtypoints"
        ] = repair_trusted_ret["errors_per_feature"][4].cpu()

        df_errors_repair_ts.to_csv(folder_output + "/trusted_error_repair_features.csv")


# Running Options:
#
#
#


def main(args):

    # NOTE: use flag: --semi-supervise for now, then make model dependent?

    # Load datasets

    # train
    (
        train_loader,
        X_train,
        target_errors_train,
        dataset_obj,
        attributes,
        trusted_mask,
    ) = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=True,
        get_data_idxs=True,
        semi_sup_data=True,
        use_binary_img=args.use_binary_img,
        trust_set_name=args.trust_set_name,
    )

    train_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_obj, batch_size=args.batch_size, shuffle=False
    )

    # validation
    (
        valid_loader,
        X_valid,
        target_errors_valid,
        dataset_valid_obj,
        _,
    ) = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=False,
        use_binary_img=args.use_binary_img,
    )

    valid_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_valid_obj, batch_size=args.batch_size, shuffle=False
    )

    # test
    test_loader, X_test, target_errors_test, dataset_test_obj, _ = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=False,
        use_binary_img=args.use_binary_img,
    )

    test_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_test_obj, batch_size=args.batch_size, shuffle=False
    )

    # -> clean versions for evaluation
    (
        train_clean_loader,
        X_train_clean,
        _,
        dataset_obj_train_clean,
        _,
    ) = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=True,
        is_clean=True,
        stdize_dirty=True,
        use_binary_img=args.use_binary_img,
    )

    train_clean_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_obj_train_clean, batch_size=args.batch_size, shuffle=False
    )

    _, X_valid_clean, _, dataset_obj_valid_clean, _ = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=False,
        is_clean=True,
        stdize_dirty=True,
        use_binary_img=args.use_binary_img,
    )

    valid_clean_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_obj_valid_clean, batch_size=args.batch_size, shuffle=False
    )

    _, X_test_clean, _, dataset_obj_test_clean, _ = gen_utils.load_data(
        args.data_folder,
        args.batch_size,
        is_train=False,
        is_clean=True,
        stdize_dirty=True,
        use_binary_img=args.use_binary_img,
    )

    test_clean_loader_no_shuff = torch.utils.data.DataLoader(
        dataset_obj_test_clean, batch_size=args.batch_size, shuffle=False
    )

    # if runnin on gpu, then load data there
    # TODO: Account for large datasets (big image datasets might overload GPU MEM)
    if args.cuda_on:
        X_train = X_train.cuda()
        X_valid = X_valid.cuda()
        X_test = X_test.cuda()

        target_errors_train = target_errors_train.cuda()
        target_errors_valid = target_errors_valid.cuda()
        target_errors_test = target_errors_test.cuda()

        X_train_clean = X_train_clean.cuda()
        X_valid_clean = X_valid_clean.cuda()
        X_test_clean = X_test_clean.cuda()

        trusted_mask = trusted_mask.cuda()

    # if supervised loss uses
    if args.use_sup_weights and args.semi_supervise:
        _num_outliers_ts = (trusted_mask * target_errors_train.any(dim=1)).sum()
        _num_inliers_ts = (
            trusted_mask * torch.logical_not(target_errors_train.any(dim=1))
        ).sum()
        args.qy_sup_weights = [
            1.0,
            max(1.0, (_num_inliers_ts / _num_outliers_ts).item()),
        ]

    else:
        args.qy_sup_weights = None

    # Import model from the correct file
    runin_model = __import__(args.model_type)
    model = runin_model.VAE(dataset_obj, args)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))

    print(args)

    if args.cuda_on:
        model.cuda()

    train_optim = StandardTrainer(
        model, args, lr_opt=args.lr, weight_decay_opt=args.l2_reg
    )

    # structs for saving data
    losses_save = {
        "train": {},
        "validation": {},
        "test": {},
        "trusted": {},
        "train_per_feature": {},
        "validation_per_feature": {},
        "test_per_feature": {},
        "trusted_per_feature": {},
    }

    # KL annealing scheduling
    kl_anneal = args.kl_anneal
    kl_beta_n_cycles = args.kl_anneal_cycles
    kl_beta_ratio = args.kl_anneal_ratio  # 0.75; 0.25

    delay_n_epochs = args.kl_anneal_delay_epochs

    if kl_anneal and args.number_epochs > delay_n_epochs:
        if delay_n_epochs > 0:
            delay_beta_vec = np.zeros(delay_n_epochs)  # 0.0  # 0.001  # 1e-6
            _delay_n_epochs = delay_n_epochs
        else:
            delay_beta_vec = []
            _delay_n_epochs = 0

        kl_beta_vec = frange_cycle_linear(
            args.kl_anneal_start,
            args.kl_anneal_stop,
            args.number_epochs - _delay_n_epochs,
            n_cycle=kl_beta_n_cycles,
            ratio=kl_beta_ratio,
        )

        kl_beta_vec = np.concatenate((delay_beta_vec, kl_beta_vec))

    else:
        kl_beta_vec = np.ones(args.number_epochs) * args.kl_beta_const
        # 1.0, 0.0; 0.001;

    print(kl_beta_vec)

    # Regularizer scheduling
    if args.dist_corr_reg and args.number_epochs > args.reg_delay_n_epochs:

        if args.reg_delay_n_epochs > 0:
            delay_reg_vec = np.zeros(args.reg_delay_n_epochs)
            _delay_n_epochs = args.reg_delay_n_epochs
        else:
            delay_reg_vec = []
            _delay_n_epochs = 0

        reg_schedule_vec = frange_cycle_linear(
            1e-6,
            1.0,
            args.number_epochs - _delay_n_epochs,
            n_cycle=1,
            ratio=args.reg_schedule_ratio,
        )

        reg_schedule_vec = np.concatenate((delay_reg_vec, reg_schedule_vec))

    else:
        reg_schedule_vec = np.ones(args.number_epochs)

    print(reg_schedule_vec)

    # option: train on clean data instead (e.g. for testing "compression hypothesis")
    if args.train_on_clean_data:
        _train_loader_used = train_clean_loader
        _train_loader_no_shuff = train_clean_loader_no_shuff
        _valid_loader_no_shuff = valid_clean_loader_no_shuff
        _test_loader_no_shuff = test_clean_loader_no_shuff
        _X_train = X_train_clean
        _X_valid = X_valid_clean
        _X_test = X_test_clean

    else:
        # standard
        _train_loader_used = train_loader
        _train_loader_no_shuff = train_loader_no_shuff
        _valid_loader_no_shuff = valid_loader_no_shuff
        _test_loader_no_shuff = test_loader_no_shuff
        _X_train = X_train
        _X_valid = X_valid
        _X_test = X_test

    # Run epochs
    for epoch in range(1, args.number_epochs + 1):

        kl_beta_val = kl_beta_vec[epoch - 1]
        reg_schedule_val = reg_schedule_vec[epoch - 1]

        print(kl_beta_val)

        print(reg_schedule_val)

        ## Train Phase
        training_phase(
            model,
            train_optim,
            _train_loader_used,  # train_loader (done)
            args,
            epoch,
            trusted_mask,
            kl_beta_val,
            reg_schedule_val,
        )

        # Compute losses and metrics per epoch
        compute_metrics(
            model,
            _train_loader_no_shuff,
            _X_train,
            dataset_obj,
            args,
            epoch,
            losses_save,
            X_train_clean,
            target_errors_train,
            trusted_mask,
            mode="train",
            kl_beta=kl_beta_val,
            reg_scheduler_val=reg_schedule_val,
        )

        ## Validation Phase
        compute_metrics(
            model,
            _valid_loader_no_shuff,
            _X_valid,
            dataset_valid_obj,
            args,
            epoch,
            losses_save,
            X_valid_clean,
            target_errors_valid,
            None,
            mode="validation",
            kl_beta=kl_beta_val,
            reg_scheduler_val=reg_schedule_val,
        )

        ## Test Phase
        compute_metrics(
            model,
            _test_loader_no_shuff,
            _X_test,
            dataset_test_obj,
            args,
            epoch,
            losses_save,
            X_test_clean,
            target_errors_test,
            None,
            mode="test",
            kl_beta=kl_beta_val,
            reg_scheduler_val=reg_schedule_val,
        )

    # save to folder AVPR / AUC per feature
    if args.save_on:

        # create folder for saving experiment data (if necessary)
        folder_output = args.output_folder + args.model_type  #  "/" +

        ### Train Data
        save_to_csv(
            model,
            _train_loader_no_shuff,
            _X_train,
            X_train_clean,
            target_errors_train,
            trusted_mask,
            attributes,
            losses_save,
            dataset_obj,
            folder_output,
            args,
            mode="train",
            kl_beta=kl_beta_vec[-1],
            reg_scheduler_val=reg_schedule_vec[-1],
        )

        ### Validation Data
        save_to_csv(
            model,
            _valid_loader_no_shuff,
            _X_valid,
            X_valid_clean,
            target_errors_valid,
            None,
            attributes,
            losses_save,
            dataset_valid_obj,
            folder_output,
            args,
            mode="validation",
            kl_beta=kl_beta_vec[-1],
            reg_scheduler_val=reg_schedule_vec[-1],
        )

        ### Test Data
        save_to_csv(
            model,
            _test_loader_no_shuff,
            _X_test,
            X_test_clean,
            target_errors_test,
            None,
            attributes,
            losses_save,
            dataset_test_obj,
            folder_output,
            args,
            mode="test",
            kl_beta=kl_beta_vec[-1],
            reg_scheduler_val=reg_schedule_vec[-1],
        )

        # save model parameters
        model.cpu()
        torch.save(model.state_dict(), folder_output + "/model_params.pth")

        # save to .json file the args that were used for running the model
        with open(folder_output + "/args_run.json", "w") as outfile:
            json.dump(vars(args), outfile, indent=4, sort_keys=True)

    return locals()  # to be used in printing / notebooks / debug


if __name__ == "__main__":

    args = parser_arguments.getArgs(sys.argv[1:])

    dict_main_vars = main(args)
