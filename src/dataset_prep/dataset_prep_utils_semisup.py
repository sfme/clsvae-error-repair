import sys

sys.path.append("..")

import pandas as pd
import numpy as np
import json
import os, errno

import copy

from collections import namedtuple

from cleaningbenchmark.NoiseModels.RandomNoise import (
    ImageSaltnPepper,
    ImageAdditiveGaussianNoise,
)

from cleaningbenchmark.Utils.Utils import pd_df_diff

from cleaningbenchmark.SyntheticData.SynthGaussianClusters import (
    SynthGaussianClusters as SGClusters,
)

from cleaningbenchmark.SyntheticData.SimpleShapesWithStripes import (
    SimpleShapesWithStripes as ShapesNStripes,
)

from cleaningbenchmark.NoiseModels.SystematicNoise import ImageSystematicSimpleShapes

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def create_data_folders(run_stats, path_to_folder):

    """ create folders """

    # path to folder where to save to
    path_saving = path_to_folder + run_stats["name"] + "/"

    # try to create folder if not exists yet
    try:
        os.makedirs(path_saving)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for full dirty dataset
    try:
        os.makedirs(path_saving + "/full/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for train dataset
    try:
        os.makedirs(path_saving + "/train/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for validation dataset
    try:
        os.makedirs(path_saving + "/validation/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for test dataset
    try:
        os.makedirs(path_saving + "/test/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return path_saving


def create_data_splits(run_stats, data, test_given=False):

    if not test_given:
        cond_test = (
            run_stats["train_size"] + run_stats["valid_size"] + run_stats["test_size"]
        ) == 1.0
        assert cond_test, "dataset size percentages (train; valid; test) must match!"

    splitter = ShuffleSplit(
        n_splits=1, test_size=(1.0 - run_stats["train_size"])
    )  # ,random_state=1 #TODO: apply Random_State?
    train_idxs, test_idxs = [x for x in splitter.split(data)][0]

    if test_given:
        validation_idxs = test_idxs
        return train_idxs, validation_idxs, None
    else:
        test_size_prop = float(run_stats["test_size"]) / (
            run_stats["valid_size"] + run_stats["test_size"]
        )

        splitter_cv = ShuffleSplit(
            n_splits=1, test_size=test_size_prop
        )  # ,random_state=1
        rel_valid_idxs, rel_test_idxs = [x for x in splitter_cv.split(test_idxs)][0]

        validation_idxs = test_idxs[rel_valid_idxs]
        test_idxs = test_idxs[rel_test_idxs]

        return train_idxs, validation_idxs, test_idxs


def _get_clean_idxs(y_noise):
    """ get idxs for clean datapoints; y_noise is binary vector """

    return np.argwhere(np.logical_not(y_noise.ravel())).flatten().tolist()


def _get_noise_idxs(y_noise):
    """ get idxs for dirty/noised datapoints; y_noise is binary vector """

    return np.argwhere(y_noise.ravel()).flatten().tolist()


def _get_y_class_lists(y_class):

    """ y_class is categorical tag (numerical) vector """

    y_class_lists = [[] for k in range(max(y_class) + 1)]

    for idx, class_tag in enumerate(y_class):
        y_class_lists[class_tag].append(idx)

    return y_class_lists


def _get_y_class_clean_lists(y_class, y_noise):

    """ y_noise is binary vector, y_class is categorical tag (numerical) vector """

    y_class_lists = _get_y_class_lists(y_class)
    y_class_lists_clean = [[] for k in range(max(y_class) + 1)]

    y_clean_idxs = _get_clean_idxs(y_noise)

    for class_tag, idx_list in enumerate(y_class_lists):
        for idx in idx_list:
            if idx in y_clean_idxs:
                y_class_lists_clean[class_tag].append(idx)

    return y_class_lists_clean


def _get_y_lists_joined(y_class_lists_clean, y_noise_lists):

    return y_class_lists_clean + y_noise_lists


def _get_y_collapsed(y_lists, n_datapoints):
    # NOTE: Can be problematic if a datapoint belongs to several classes (e.g. usually noise types, not actual underlying dataset classes)
    #       Thus may fail guaranteed minimum coverage of class, or proper uniform coverage!

    # NOTE: For general case, to resolve above issue, use y_lists directly instead, e.g. stratisfied_v2!

    # NOTE: Works fine in the case that each datapoint has a single noise_label; even better if all datapoints labelled!

    y_lists_collapsed = np.ones(n_datapoints, dtype=int) * -1

    for ii, class_idxs in enumerate(y_lists):
        for idx in class_idxs:
            y_lists_collapsed[idx] = ii

    if (
        y_lists_collapsed == -1
    ).any():  # for datapoints that have not been labelled (assign some class)
        y_lists_collapsed[y_lists_collapsed == -1] = max(y_lists_collapsed) + 1

    return y_lists_collapsed


def _filter_y_list(y_lists, filter_idxs):

    return [[idx for idx in cur_list if idx in filter_idxs] for cur_list in y_lists]


def _trf_idxs_split_y_lists(y_lists, filter_idxs):
    """ transform original idxs into current split idxs """

    idx_map = dict(zip(filter_idxs, range(len(filter_idxs))))

    y_list_distil = _filter_y_list(y_lists, filter_idxs)

    for class_list in y_list_distil:
        for elem, idx_orig in enumerate(class_list):
            class_list[elem] = int(idx_map[idx_orig])

    return y_list_distil


def _list_cast_int(list_ints):

    """ due to numpy ints not being serializable """

    return [int(val_int) for val_int in list_ints]


def _save_ll_to_dl(y_lls):
    """ save from list-of-lists, to dict-of-lists """

    return {
        key_class: _list_cast_int(cur_list) for key_class, cur_list in enumerate(y_lls)
    }


def _combine_y_lists(y_lists_1, y_lists_2, offset=-1):

    if offset > 0:
        _y_l_2 = [[elem + offset for elem in elem_list] for elem_list in y_lists_2]
    else:
        _y_l_2 = y_lists_2

    _y_l_1 = copy.deepcopy(y_lists_1)

    for ix, elem_list in enumerate(_y_l_1):
        elem_list.extend(_y_l_2[ix])

    return _y_l_1


def _enforce_data_split(
    df_data,
    train_idxs,
    validation_idxs,
    test_idxs,
    df_noise_data,
    cells_changed,
    tuples_changed,
    y_class=None,
    y_noise_lists=None,
    y_noise_lists_combs=None,
    test_set_in=None,
):

    train_dataset = dict()
    valid_dataset = dict()
    test_dataset = dict()

    ## train dataset
    train_dataset["og_idxs"] = train_idxs

    # clean data
    df_train = df_data.iloc[train_idxs, :]
    df_train = df_train.reset_index(drop=True)
    train_dataset["df_data"] = df_train

    # dirty data
    df_train_noised = df_noise_data.iloc[train_idxs, :]
    df_train_noised = df_train_noised.reset_index(drop=True)
    train_dataset["df_noise_data"] = df_train_noised

    # ground-truth of errors - error matrix (binary)
    train_dataset["cells_changed_oh"] = cells_changed[train_idxs, :]

    # ground-truth of errors - rows with errors (binary)
    train_dataset["tuples_changed_oh"] = tuples_changed[train_idxs]

    ## validation dataset
    valid_dataset["og_idxs"] = validation_idxs

    # clean data
    df_validation = df_data.iloc[validation_idxs, :]
    df_validation = df_validation.reset_index(drop=True)
    valid_dataset["df_data"] = df_validation

    # dirty data
    df_validation_noised = df_noise_data.iloc[validation_idxs, :]
    df_validation_noised = df_validation_noised.reset_index(drop=True)
    valid_dataset["df_noise_data"] = df_validation_noised

    # ground-truth of errors - error matrix (binary)
    valid_dataset["cells_changed_oh"] = cells_changed[validation_idxs, :]

    # ground-truth of errors - rows with errors (binary)
    valid_dataset["tuples_changed_oh"] = tuples_changed[validation_idxs]

    ## test dataset
    if test_idxs is not None:
        test_dataset["og_idxs"] = test_idxs

        # clean data
        df_test = df_data.iloc[test_idxs, :]
        df_test = df_test.reset_index(drop=True)
        test_dataset["df_data"] = df_test

        # dirty data
        df_test_noised = df_noise_data.iloc[test_idxs, :]
        df_test_noised = df_test_noised.reset_index(drop=True)
        test_dataset["df_noise_data"] = df_test_noised

        # ground-truth of errors - error matrix (binary)
        test_dataset["cells_changed_oh"] = cells_changed[test_idxs, :]

        # ground-truth of errors - rows with errors (binary)
        test_dataset["tuples_changed_oh"] = tuples_changed[test_idxs]
    else:
        test_dataset["og_idxs"] = None

        # clean data
        test_dataset["df_data"] = test_set_in["df_data"]

        # dirty data
        test_dataset["df_noise_data"] = test_set_in["df_noise_data"]

        # ground-truth of errors - error matrix (binary)
        test_dataset["cells_changed_oh"] = test_set_in["cells_changed_oh"]

        # ground-truth of errors - rows with errors (binary)
        test_dataset["tuples_changed_oh"] = test_set_in["tuples_changed_oh"]

    if y_class is not None:
        train_dataset["y_class"] = y_class[train_idxs]
        valid_dataset["y_class"] = y_class[validation_idxs]
        if test_idxs is not None:
            test_dataset["y_class"] = y_class[test_idxs]
        else:
            test_dataset["y_class"] = test_set_in["y_class"]

    else:
        train_dataset["y_class"] = None
        valid_dataset["y_class"] = None
        test_dataset["y_class"] = None

    if y_noise_lists is not None:
        train_dataset["y_noise_lists"] = _trf_idxs_split_y_lists(
            y_noise_lists, train_idxs
        )
        valid_dataset["y_noise_lists"] = _trf_idxs_split_y_lists(
            y_noise_lists, validation_idxs
        )
        if test_idxs is not None:
            test_dataset["y_noise_lists"] = _trf_idxs_split_y_lists(
                y_noise_lists, test_idxs
            )
        else:
            test_dataset["y_noise_lists"] = test_set_in["y_noise_lists"]
    else:
        train_dataset["y_noise_lists"] = None
        valid_dataset["y_noise_lists"] = None
        test_dataset["y_noise_lists"] = None

    if y_noise_lists_combs is not None:
        train_dataset["y_noise_lists_combs"] = _trf_idxs_split_y_lists(
            y_noise_lists_combs, train_idxs
        )
        valid_dataset["y_noise_lists_combs"] = _trf_idxs_split_y_lists(
            y_noise_lists_combs, validation_idxs
        )
        if test_idxs is not None:
            test_dataset["y_noise_lists_combs"] = _trf_idxs_split_y_lists(
                y_noise_lists_combs, test_idxs
            )
        else:
            test_dataset["y_noise_lists_combs"] = test_set_in["y_noise_lists_combs"]
    else:
        train_dataset["y_noise_lists_combs"] = None
        valid_dataset["y_noise_lists_combs"] = None
        test_dataset["y_noise_lists_combs"] = None

    return train_dataset, valid_dataset, test_dataset


def _save_datasets(
    path_saving,
    train_dataset,
    valid_dataset,
    test_dataset,
    full_dataset,
    trusted_idxs,
    trust_set_type=None,
):

    # NOTE:
    # synth_train_dataset['df_data']
    # synth_train_dataset['df_noise_data']
    # synth_train_dataset['cells_changed_oh']
    # synth_train_dataset['tuples_changed_oh']
    # synth_train_dataset['y_class']
    # synth_train_dataset['y_noise_lists']
    # synth_train_dataset['og_idxs']

    ## train

    # ground-truth of errors - dataframe value changes
    df_changes_train, _, _ = pd_df_diff(
        train_dataset["df_data"], train_dataset["df_noise_data"]
    )

    train_dataset["df_data"].to_csv(
        path_saving + "/train/" + "data_clean.csv", index=False
    )
    train_dataset["df_noise_data"].to_csv(
        path_saving + "/train/" + "data_noised.csv", index=False
    )
    df_changes_train.to_csv(path_saving + "/train/" + "changes_summary.csv")

    df_cells_changed_train = pd.DataFrame(
        train_dataset["cells_changed_oh"], columns=full_dataset["df_data"].columns
    )
    df_cells_changed_train.to_csv(
        path_saving + "/train/" + "cells_changed_mtx.csv", index=False
    )

    df_tuples_changed_train = pd.DataFrame(
        train_dataset["tuples_changed_oh"], columns=["rows_with_outlier"]
    )
    df_tuples_changed_train.to_csv(
        path_saving + "/train/" + "tuples_changed_mtx.csv", index=False
    )

    df_train_idxs = pd.DataFrame(train_dataset["og_idxs"], columns=["original_idxs"])
    df_train_idxs.to_csv(path_saving + "/train/" + "original_idxs.csv", index=False)

    # store ground-truth labels for classes
    if train_dataset["y_class"] is not None:
        df_train_y_class = pd.DataFrame(
            train_dataset["y_class"], columns=["class_labels"]
        )
        df_train_y_class.to_csv(path_saving + "/train/" + "y_class.csv", index=False)

    if train_dataset["y_noise_lists"] is not None:
        dl_train_noise = _save_ll_to_dl(train_dataset["y_noise_lists"])
        with open(path_saving + "/train/" + "y_noise_dict.json", "w") as outfile:
            json.dump(dl_train_noise, outfile, indent=4, sort_keys=True)

    # store labelled trusted set idxs for loading after
    if isinstance(trusted_idxs, dict):
        _ts_dict = trusted_idxs

        for ts_cur_quant, ts_cur_idxs in _ts_dict.items():

            if trust_set_type == "samples":
                _file_out = (
                    f"trusted_idxs_{int(ts_cur_quant)}_{trust_set_type}_per_class.csv"
                )
            elif trust_set_type == "percent":
                _ts_temp = ts_cur_quant * 100.0
                _file_out = f"trusted_idxs_{_ts_temp:.4f}_{trust_set_type}.csv"

            _df_ts_idxs = pd.DataFrame(ts_cur_idxs, columns=["labels_idxs"])
            _df_ts_idxs.to_csv(
                path_saving + "/train/" + _file_out,
                index=False,
            )

    else:
        df_train_trust_idxs = pd.DataFrame(trusted_idxs, columns=["labels_idxs"])
        df_train_trust_idxs.to_csv(
            path_saving + "/train/" + "trusted_idxs.csv", index=False
        )

    ## validation

    # ground-truth of errors - dataframe value changes
    df_changes_validation, _, _ = pd_df_diff(
        valid_dataset["df_data"], valid_dataset["df_noise_data"]
    )

    valid_dataset["df_data"].to_csv(
        path_saving + "/validation/" + "data_clean.csv", index=False
    )
    valid_dataset["df_noise_data"].to_csv(
        path_saving + "/validation/" + "data_noised.csv", index=False
    )
    df_changes_validation.to_csv(path_saving + "/validation/" + "changes_summary.csv")

    df_cells_changed_validation = pd.DataFrame(
        valid_dataset["cells_changed_oh"], columns=full_dataset["df_data"].columns
    )
    df_cells_changed_validation.to_csv(
        path_saving + "/validation/" + "cells_changed_mtx.csv", index=False
    )

    df_tuples_changed_validation = pd.DataFrame(
        valid_dataset["tuples_changed_oh"], columns=["rows_with_outlier"]
    )
    df_tuples_changed_validation.to_csv(
        path_saving + "/validation/" + "tuples_changed_mtx.csv", index=False
    )

    df_validation_idxs = pd.DataFrame(
        valid_dataset["og_idxs"], columns=["original_idxs"]
    )
    df_validation_idxs.to_csv(
        path_saving + "/validation/" + "original_idxs.csv", index=False
    )

    # store ground-truth labels for classes
    if valid_dataset["y_class"] is not None:
        df_valid_y_class = pd.DataFrame(
            valid_dataset["y_class"], columns=["class_labels"]
        )
        df_valid_y_class.to_csv(
            path_saving + "/validation/" + "y_class.csv", index=False
        )

    if valid_dataset["y_noise_lists"] is not None:
        dl_valid_noise = _save_ll_to_dl(valid_dataset["y_noise_lists"])
        with open(path_saving + "/validation/" + "y_noise_dict.json", "w") as outfile:
            json.dump(dl_valid_noise, outfile, indent=4, sort_keys=True)

    ## test

    # ground-truth of errors - dataframe value changes
    df_changes_test, _, _ = pd_df_diff(
        test_dataset["df_data"], test_dataset["df_noise_data"]
    )

    test_dataset["df_data"].to_csv(
        path_saving + "/test/" + "data_clean.csv", index=False
    )
    test_dataset["df_noise_data"].to_csv(
        path_saving + "/test/" + "data_noised.csv", index=False
    )
    df_changes_test.to_csv(path_saving + "/test/" + "changes_summary.csv")

    df_cells_changed_test = pd.DataFrame(
        test_dataset["cells_changed_oh"], columns=full_dataset["df_data"].columns
    )
    df_cells_changed_test.to_csv(
        path_saving + "/test/" + "cells_changed_mtx.csv", index=False
    )

    df_tuples_changed_test = pd.DataFrame(
        test_dataset["tuples_changed_oh"], columns=["rows_with_outlier"]
    )
    df_tuples_changed_test.to_csv(
        path_saving + "/test/" + "tuples_changed_mtx.csv", index=False
    )

    if test_dataset["og_idxs"] is not None:
        df_test_idxs = pd.DataFrame(test_dataset["og_idxs"], columns=["original_idxs"])
        df_test_idxs.to_csv(path_saving + "/test/" + "original_idxs.csv", index=False)

    # store ground-truth labels for classes
    if test_dataset["y_class"] is not None:
        df_test_y_class = pd.DataFrame(
            test_dataset["y_class"], columns=["class_labels"]
        )
        df_test_y_class.to_csv(path_saving + "/test/" + "y_class.csv", index=False)

    if test_dataset["y_noise_lists"] is not None:
        dl_test_noise = _save_ll_to_dl(test_dataset["y_noise_lists"])
        with open(path_saving + "/test/" + "y_noise_dict.json", "w") as outfile:
            json.dump(dl_test_noise, outfile, indent=4, sort_keys=True)

    ## full

    # ground-truth of errors
    df_changes_full, _, _ = pd_df_diff(
        full_dataset["df_data"], full_dataset["df_noise_data"]
    )

    # save
    full_dataset["df_data"].to_csv(
        path_saving + "/full/" + "data_clean.csv", index=False
    )
    full_dataset["df_noise_data"].to_csv(
        path_saving + "/full/" + "data_noised.csv", index=False
    )
    df_changes_full.to_csv(path_saving + "/full/" + "changes_summary.csv")

    df_cells_changed = pd.DataFrame(
        full_dataset["cells_changed_oh"], columns=full_dataset["df_data"].columns
    )
    df_cells_changed.to_csv(
        path_saving + "/full/" + "cells_changed_mtx.csv", index=False
    )

    df_tuples_changed = pd.DataFrame(
        full_dataset["tuples_changed_oh"], columns=["rows_with_outlier"]
    )
    df_tuples_changed.to_csv(
        path_saving + "/full/" + "tuples_changed_mtx.csv", index=False
    )

    # store ground-truth labels for classes
    if full_dataset["y_class"] is not None:
        df_full_y_class = pd.DataFrame(
            full_dataset["y_class"], columns=["class_labels"]
        )
        df_full_y_class.to_csv(path_saving + "/full/" + "y_class.csv", index=False)

    if full_dataset["y_noise_lists"] is not None:
        dl_full_noise = _save_ll_to_dl(full_dataset["y_noise_lists"])
        with open(path_saving + "/full/" + "y_noise_dict.json", "w") as outfile:
            json.dump(dl_full_noise, outfile, indent=4, sort_keys=True)


def create_trust_set_split(
    run_stats,
    data_train,
    label_type="no_labels",
    y_class=None,
    y_noise=None,
    y_noise_lists=None,
):

    """y_class (class tags: 0,..,C) from underlying dataset, y_noise (binary - 0,1); both numpy arrays; both are np.arrays.

    label_type (str) -- a) clean_class_only (from y_class, only clean data);
                        b) dirty_classes_only (from y_noise_lists, only dirty data);
                        c) joint_classes --> a) and b) together (both dirty and clean);
                        d) no_labels --> randomly pick a trusted set (ignore labels when sampling trusted set);

        ^ this option appears in 'run_stats['trusted_set']['use_labels']'

    NOTE: -- i) If no y_noise_lists, then only one class of dirty datapoint (e.g. the case of random errors)
          -- ii) If no y_class, then only one class of clean datapoint (e.g. does not know underlying dataset classes)
          -- iii) usually y_noise is given and known, but one can not provide it, needed for any label-based modes!
    """

    if label_type == "clean_class_only":  # option a)
        if (y_class is not None) and (y_noise is not None):
            y_class_lists_clean = _get_y_class_clean_lists(y_class, y_noise)
        elif (y_class is None) and (
            y_noise is not None
        ):  # see note above ii) (one class -- the clean labels)
            y_class_lists_clean = [_get_clean_idxs(y_noise)]
        else:
            raise ValueError(
                "Wrong values for arguments provided (not None), see:",
                "y_class" + str(y_class),
                "y_noise" + str(y_noise),
            )
        y_collapsed_lists = y_class_lists_clean

    elif label_type == "dirty_classes_only":  # option b)
        if (y_noise is not None) and (y_noise_lists is not None):
            y_class_lists_dirty = y_noise_lists
        elif (y_noise is not None) and (
            y_noise_lists is None
        ):  # see note above i) (one class -- the dirty labels)
            y_class_lists_dirty = [_get_noise_idxs(y_noise)]
        else:
            raise ValueError(
                "Wrong values for arguments provided (not None), see:",
                "y_noise" + str(y_noise),
                "y_noise_lists" + str(y_noise_lists),
            )
        y_collapsed_lists = y_class_lists_dirty

    elif label_type == "joint_classes":  # option a) + b)
        if y_noise is not None:
            if y_class is None:
                y_class_lists_clean = [_get_clean_idxs(y_noise)]
            else:
                y_class_lists_clean = _get_y_class_clean_lists(y_class, y_noise)

            if y_noise_lists is None:
                y_class_lists_dirty = [_get_noise_idxs(y_noise)]
            else:
                y_class_lists_dirty = y_noise_lists

            y_collapsed_lists = _get_y_lists_joined(
                y_class_lists_clean, y_class_lists_dirty
            )

        else:
            raise ValueError(
                "Wrong values for arguments provided (not None), see:",
                "y_class" + str(y_class),
                "y_noise" + str(y_noise),
                "y_noise_lists" + str(y_noise_lists),
            )

    else:
        if label_type != "no_labels":  # then must be option d)
            raise ValueError("Labelling option for trusted set not available.")

    # for reuse of max size trusted set on smaller trust sets
    trust_set_splits = dict()

    ## get indexes for trusted set creation
    if not run_stats["trusted_set"]["min_coverage"] and label_type == "no_labels":
        # fully random, does not guarantee minimum coverage of classes

        if isinstance(run_stats["trusted_set"]["frac_trusted"], (list, np.ndarray)):

            _max_frac_size = np.max(run_stats["trusted_set"]["frac_trusted"])

            splitter = ShuffleSplit(
                n_splits=1, test_size=_max_frac_size
            )  # ,random_state=?
            _, trust_set_idxs = list(splitter.split(data_train))[0]

            _data_len = data_train.shape[0]

            trust_set_splits = dict()
            for ts_frac in run_stats["trusted_set"]["frac_trusted"]:
                _tmp_n_samples = max(int(ts_frac * _data_len), trust_set_idxs.size)
                trust_set_splits[ts_frac] = trust_set_idxs[:_tmp_n_samples]

        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=run_stats["trusted_set"]["frac_trusted"]
            )  # ,random_state=?
            _, trust_set_idxs = list(splitter.split(data_train))[
                0
            ]  # ignores any labels y

    elif run_stats["trusted_set"]["min_coverage"] and label_type != "no_labels":
        # advanced options, that may guarantee minimum coverage of classes
        if run_stats["trusted_set"]["mc_mode"] == "stratisfied":
            # NOTE: only for single use, not array.
            if isinstance(run_stats["trusted_set"]["frac_trusted"], (list, np.ndarray)):
                raise ValueError(
                    "Only for single fraction of trust set use, not array or list! Use stratisfied_v2!"
                )

            # stratisfied sampling per class
            y_collapsed = _get_y_collapsed(
                y_collapsed_lists, data_train.shape[0]
            )  # NOTE: see exceptions of function!
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=run_stats["trusted_set"]["frac_trusted"]
            )  # ,random_state=?
            _, trust_set_idxs = list(splitter.split(data_train, y_collapsed))[0]

        elif run_stats["trusted_set"]["mc_mode"] == "fixed_number":
            # a certain number of samples per class (no percentage)
            if isinstance(
                run_stats["trusted_set"]["samples_fixed"], (list, np.ndarray)
            ):

                _max_ts_samples = int(np.max(run_stats["trusted_set"]["samples_fixed"]))

                trust_set_idxs = []
                trust_set_idxs_lists = []
                for class_idx_list in y_collapsed_lists:

                    _aux = [
                        cur_idx
                        for cur_idx in class_idx_list
                        if cur_idx not in trust_set_idxs
                    ]  # filter out already used idxs (e.g. np.unique)

                    _aux = np.random.choice(_aux, size=_max_ts_samples, replace=False)
                    trust_set_idxs.extend(_aux.tolist())
                    trust_set_idxs_lists.append(_aux.tolist())
                trust_set_idxs = np.array(trust_set_idxs)

                trust_set_splits = dict()
                for ts_size in run_stats["trusted_set"]["samples_fixed"]:
                    ts_size = int(ts_size)

                    _temp_list = []
                    for ts_idx_list in trust_set_idxs_lists:
                        _temp_list.append(ts_idx_list[:ts_size])

                    trust_set_splits[ts_size] = np.concatenate(_temp_list)

            else:
                trust_set_idxs = []
                for class_idx_list in y_collapsed_lists:
                    # NOTE: may crash if too many points are asked, but not enough exist in said class.
                    _aux = [
                        cur_idx
                        for cur_idx in class_idx_list
                        if cur_idx not in trust_set_idxs
                    ]  # filter out already used idxs (e.g. np.unique)

                    _aux = np.random.choice(
                        _aux,
                        size=run_stats["trusted_set"]["samples_fixed"],
                        replace=False,
                    )
                    trust_set_idxs.extend(_aux.tolist())
                trust_set_idxs = np.array(trust_set_idxs)

        elif run_stats["trusted_set"]["mc_mode"] == "stratisfied_v2":
            # stratisfied sampling per class (uses classes list instead for more control)
            # NOTE: only truly stratisfied if every single datapoint in trainset has been labelled (i.e. accounted for in a group)!

            if isinstance(run_stats["trusted_set"]["frac_trusted"], (list, np.ndarray)):

                _max_ts_fraction = np.max(run_stats["trusted_set"]["frac_trusted"])

                trust_set_idxs = []
                trust_set_idxs_lists = []
                for class_idx_list in y_collapsed_lists:
                    n_idxs_select = int(
                        np.around(_max_ts_fraction * len(class_idx_list))
                    )

                    _aux = [
                        cur_idx
                        for cur_idx in class_idx_list
                        if cur_idx not in trust_set_idxs
                    ]  # filter out already used idxs (e.g. np.unique)

                    _aux = np.random.choice(_aux, size=n_idxs_select, replace=False)
                    trust_set_idxs.extend(_aux.tolist())
                    trust_set_idxs_lists.append(_aux.tolist())
                trust_set_idxs = np.array(trust_set_idxs)

                trust_set_splits = dict()
                for ts_frac in run_stats["trusted_set"]["frac_trusted"]:

                    _temp_list = []
                    for ts_idx_list, all_idx_list in zip(
                        trust_set_idxs_lists, y_collapsed_lists
                    ):
                        n_idxs_select = max(
                            int(np.around(ts_frac * len(all_idx_list))),
                            len(ts_idx_list),
                        )

                        _temp_list.append(ts_idx_list[:n_idxs_select])

                    trust_set_splits[ts_frac] = np.concatenate(_temp_list)

            else:
                trust_set_idxs = []
                for class_idx_list in y_collapsed_lists:
                    n_idxs_select = int(
                        np.around(
                            run_stats["trusted_set"]["frac_trusted"]
                            * len(class_idx_list)
                        )
                    )

                    _aux = [
                        cur_idx
                        for cur_idx in class_idx_list
                        if cur_idx not in trust_set_idxs
                    ]  # filter out already used idxs (e.g. np.unique)

                    _aux = np.random.choice(_aux, size=n_idxs_select, replace=False)
                    trust_set_idxs.extend(_aux.tolist())
                trust_set_idxs = np.array(trust_set_idxs)

        elif run_stats["trusted_set"]["mc_mode"] == "top_k_classes":
            raise ValueError("Option not implemented yet!!")

        else:
            raise ValueError(
                "Option does not exist, or or not enough inputs provided !!"
            )

    else:
        raise ValueError(
            "Option for trusted set creation does not exist, or not enough inputs provided !!"
        )

    return trust_set_idxs, trust_set_splits


############ semi-supervised dataset versions ###############


def dirty_semi_supervised_synthethic(run_stats, path_to_folder):
    # it is defined as a per type of Synthetic Dataset

    ### create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    ### get synthethic dataset  (INLIERS + OUTLIERS)
    if run_stats["synth_data"]["type"] == "GaussianClusters":

        # Options:
        # run_stats['synth_data']['n_samples']
        # run_stats['synth_data']['n_clusters']
        # run_stats['synth_data']['corrupt_prob']
        # run_stats['synth_data']['n_features']
        # run_stats['synth_data']['scale_density']
        # run_stats['synth_data']['size_cluster']
        # run_stats['synth_data']['std_scaler_cluster']
        # run_stats['synth_data']['dist']
        # run_stats['synth_data']['random_state']
        # run_stats['synth_data']['noise_type']
        # run_stats['synth_data']['noise_type_defs']

        synth_data = SGClusters(**run_stats["synth_data"]["defs"])

        ##  Any synthetic dataset creator returns:
        # synth_data.X # dirty data matrix
        # synth_data.y_noise # binary array (0 dirty; 1 clean)
        # synth_data.X_gt # underlying clean data matrix
        # synth_data.y_class # dataset class tags (can be None)
        # synth_data.y_noise_lists # in list form (can be None)

        # NOTE: synth_data.y_class, synth_data.y_noise_lists in general must always be defined! (None if not possible / available)

        # needed for data loading and pipeline
        synth_defs = dict()
        synth_defs["cat_cols_names"] = []
        synth_defs["num_cols_names"] = [
            str(val) for val in range(synth_data.X.shape[1])
        ]
        synth_defs["dataset_type"] = "real"  # this synthetic dataset, real feats only

        # get data definitions in dataframes
        synth_clean_data = pd.DataFrame(
            synth_data.X_gt, dtype=np.float32, columns=synth_defs["num_cols_names"]
        )
        synth_noised_data = pd.DataFrame(
            synth_data.X, dtype=np.float32, columns=synth_defs["num_cols_names"]
        )

        synth_data.y_noise_lists_combs = None
        noise_list_trust = "regular"

    elif run_stats["synth_data"]["type"] == "ShapesWithStrips":

        synth_data = ShapesNStripes(**run_stats["synth_data"]["defs"])
        # NOTE: do not forget to decide if "combs_on"; and "noise_list_trusted" flags

        # needed for data loading and pipeline
        synth_defs = dict()
        synth_defs["cat_cols_names"] = []
        synth_defs["num_cols_names"] = [
            "pixel_" + str(val) for val in range(synth_data.X.shape[1])
        ]
        # NOTE: num_cols_name, allows for both quantized (e.g. binary) and real feats.
        synth_defs["dataset_type"] = "image"

        # get data definitions in dataframes
        synth_clean_data = pd.DataFrame(
            synth_data.X_gt, dtype=np.float32, columns=synth_defs["num_cols_names"]
        )
        synth_noised_data = pd.DataFrame(
            synth_data.X, dtype=np.float32, columns=synth_defs["num_cols_names"]
        )

        # image dataset defs (e.g. used by model pipeline)
        synth_defs["image_defs"] = dict()
        synth_defs["image_defs"]["size"] = (28, 28)
        synth_defs["image_defs"]["channel_type"] = "gray"  # vs. rgb
        synth_defs["image_defs"]["num_channels"] = 1
        synth_defs["image_defs"]["channels"] = []

        # define what kind of labelling for trusted set
        if (run_stats["synth_data"]["noise_list_trusted"] is None) or (
            not run_stats["synth_data"]["defs"]["combs_on"]
        ):
            run_stats["synth_data"]["noise_list_trusted"] = "regular"

        if run_stats["synth_data"]["noise_list_trusted"] not in [
            "label_combs",
            "regular",
        ]:
            raise ValueError("Option unclear. Either use: 'regular' or 'label_combs' ")
        # "label_combs": makes sure examples with combinations of errors are labelled (e.g. stratisfied).
        # "regular": usual y_noise_lists, does not guarantee or provide labels on noise combinations.

        noise_list_trust = run_stats["synth_data"]["noise_list_trusted"]

    else:
        raise ValueError("Name of Synthethic Dataset Generator does not exist.")

    _, cells_changed, _ = pd_df_diff(
        synth_clean_data, synth_noised_data
    )  # NOTE: ,tuples_changed must be equal to .y_noise

    tuples_changed = synth_data.y_noise.astype(int)
    cells_changed = cells_changed.astype(int)
    # tuples_changed = tuples_changed.astype(int)

    ## get dataset splits
    train_idxs, validation_idxs, test_idxs = create_data_splits(
        run_stats, synth_clean_data
    )

    ret = _enforce_data_split(
        synth_clean_data,
        train_idxs,
        validation_idxs,
        test_idxs,
        synth_noised_data,
        cells_changed,
        tuples_changed,
        synth_data.y_class,
        synth_data.y_noise_lists,
        synth_data.y_noise_lists_combs,
    )
    synth_train_dataset, synth_valid_dataset, synth_test_dataset = ret

    ## for instance:
    # synth_train_dataset['df_data']
    # synth_train_dataset['df_noise_data']
    # synth_train_dataset['cells_changed_oh']
    # synth_train_dataset['tuples_changed_oh']
    # synth_train_dataset['y_class']
    # synth_train_dataset['y_noise_lists']
    # synth_train_dataset['og_idxs']

    ## get trusted set (subset of training dataset, for semi-supervised labelling)
    if not run_stats["trusted_set"]["y_class_on"]:
        # NOTE: maybe flag is needed in data_synth process?
        train_y_class_in = None
    else:
        train_y_class_in = synth_train_dataset["y_class"]

    if not run_stats["trusted_set"]["y_noise_lists_on"]:
        train_y_noise_lists_in = None
    else:
        if noise_list_trust == "regular":
            train_y_noise_lists_in = synth_train_dataset["y_noise_lists"]
        elif noise_list_trust == "label_combs":
            train_y_noise_lists_in = synth_train_dataset["y_noise_lists_combs"]
            # NOTE: y_noise_lists_combs is not being saved, for trust set build mainly.

    trust_set_idxs, trust_set_dicts = create_trust_set_split(
        run_stats,
        synth_train_dataset["df_data"],
        label_type=run_stats["trusted_set"]["use_labels"],
        y_class=train_y_class_in,
        y_noise=synth_train_dataset["tuples_changed_oh"],
        y_noise_lists=train_y_noise_lists_in,
    )

    if trust_set_dicts:
        # NOTE: exists -> np.array or list for run_stats["trusted_set"] input: fraction / fixed samples
        _ts_input = trust_set_dicts

        if run_stats["trusted_set"]["mc_mode"] == "fixed_number":
            ts_type = "samples"

        elif (run_stats["trusted_set"]["mc_mode"] == "stratisfied_v2") or (
            not run_stats["trusted_set"]["min_coverage"]
        ):
            ts_type = "percent"
    else:
        _ts_input = trust_set_idxs
        ts_type = None

    # struct for entire dataset
    synth_full_dataset = dict()
    synth_full_dataset["df_data"] = synth_clean_data
    synth_full_dataset["df_noise_data"] = synth_noised_data
    synth_full_dataset["cells_changed_oh"] = cells_changed
    synth_full_dataset["tuples_changed_oh"] = tuples_changed
    synth_full_dataset["y_class"] = synth_data.y_class
    synth_full_dataset["y_noise_lists"] = synth_data.y_noise_lists

    # save to folders
    _save_datasets(
        path_saving,
        synth_train_dataset,
        synth_valid_dataset,
        synth_test_dataset,
        synth_full_dataset,
        _ts_input,
        trust_set_type=ts_type,
    )

    ## save num_cols, cat_cols names, and other defs for use by model code
    cols_info = {
        "cat_cols_names": synth_defs["cat_cols_names"],
        "num_cols_names": synth_defs["num_cols_names"],
        "dataset_type": synth_defs["dataset_type"],
    }
    if synth_defs["dataset_type"] == "image":
        cols_info["image_defs"] = synth_defs["image_defs"]

    with open(path_saving + "cols_info.json", "w") as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    with open(path_saving + "noising_info.json", "w") as outfile:
        json.dump(run_stats, outfile, indent=4, sort_keys=True)


def dirty_semi_supervised_image(
    array_data,
    run_stats,
    path_to_folder,
    image_dim_in=[28, 28],
    y_class_in=None,
    array_data_test=None,
    y_class_in_test=None,
):
    """
    array_data := must be numpy array with image data (examples, x_dim, y_dim)

    run_stats := definitions to dirtify dataset

    path_to_folder := folder where to save datasets

    y_class_in := (target labels -- underlying clean data) is given by user, 1d numpy array.

    Objective: takes real dataset (images), and injects with synthetic corruption (e.g. systematic).

    NOTE: For now only gray-scale images.
    """

    ## create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    array_data = array_data.astype(np.float32)

    if len(array_data.shape) == 2:
        array_data = array_data.reshape(-1, image_dim_in[0], image_dim_in[1])

    # helper definitions
    num_points = array_data.shape[0]
    x_dim = array_data.shape[1]
    y_dim = array_data.shape[2]
    img_num_pixels = x_dim * y_dim
    col_names = ["pixel_{}".format(n) for n in range(img_num_pixels)]

    test_set_given = False

    if array_data_test is not None:
        test_set_given = True
        _test_num_points = array_data_test.shape[0]

        if len(array_data_test.shape) == 2:
            array_data_test = array_data_test.astype(np.float32)
            array_data_test = array_data_test.reshape(
                -1, image_dim_in[0], image_dim_in[1]
            )

    # TODO: future: color images as well?
    # image dataset defs (e.g. used by model pipeline)
    image_defs = dict()
    image_defs["size"] = (x_dim, y_dim)  # (28, 28)
    image_defs["channel_type"] = "gray"  # vs. rgb
    image_defs["num_channels"] = 1
    image_defs["channels"] = []

    ## dirtify dataset

    # NOTE:
    # a) some datasets don't have y_class (or just one class) --> so None
    # b) y_noise_lists don't exist for random errors? yup (or just one class) --> so None

    NoisedDataStruct = namedtuple(
        "NoisedDataStruct", "X X_gt y_class y_noise y_noise_lists y_noise_lists_combs"
    )

    # define image (numerical) noise model
    if run_stats["type_noise"] == "SystematicSimpleShapes":
        is_random = False

        # systematic errors: using added simple shapes
        noise_mdl = ImageSystematicSimpleShapes(
            array_data.shape,
            img_prob_noise=run_stats["defs"]["p_img"],
            min_val=run_stats["defs"]["min_val"],
            max_val=run_stats["defs"]["max_val"],
            prob_min=run_stats["defs"]["p_min"],
            pixel_val_fixed=run_stats["defs"]["pixel_val_fixed"],
            number_blocks=run_stats["defs"]["number_blocks"],
            rand_blocks=run_stats["defs"]["rand_blocks"],
            side_len=run_stats["defs"]["side_len"],
            std_shift=run_stats["defs"]["std_shift"],
            use_other_patterns=run_stats["defs"]["use_other_patterns"],
            random_state=run_stats["defs"]["random_state"],
            combs_on=run_stats["defs"]["combs_on"],
        )

        X_noise, _y_noise, _y_noise_lists, _y_noise_lists_combs = noise_mdl.apply(
            array_data
        )

        noised_data_obj = NoisedDataStruct(
            X_noise.reshape((num_points, -1)),  # flatten img dims
            array_data.reshape((num_points, -1)),  # flatten img dims
            y_class_in,
            _y_noise,
            _y_noise_lists,
            _y_noise_lists_combs,
        )

        if array_data_test is not None:
            (
                X_noise_test,
                _y_noise_test,
                _y_noise_lists_test,
                _y_noise_lists_combs_test,
            ) = noise_mdl.apply(array_data_test)

            noised_data_test_obj = NoisedDataStruct(
                X_noise_test.reshape((_test_num_points, -1)),  # flatten img dims
                array_data_test.reshape((_test_num_points, -1)),  # flatten img dims
                y_class_in_test,
                _y_noise_test,
                _y_noise_lists_test,
                _y_noise_lists_combs_test,
            )

    elif run_stats["type_noise"] == "SaltnPepper":
        is_random = True

        # standard Salt and Pepper Noise
        noise_mdl = ImageSaltnPepper(
            array_data.shape,
            probability=run_stats["defs"]["p_img"],
            one_cell_flag=False,
            min_val=run_stats["defs"]["min_val"],
            max_val=run_stats["defs"]["max_val"],
            p_min=run_stats["defs"]["p_min"],
            p_pixel=run_stats["defs"]["p_pixel"],
            conv_to_int=run_stats["defs"]["conv_to_int"],
        )

        if array_data_test is not None:
            noise_mdl_test = ImageSaltnPepper(
                array_data_test.shape,
                probability=run_stats["defs"]["p_img"],
                one_cell_flag=False,
                min_val=run_stats["defs"]["min_val"],
                max_val=run_stats["defs"]["max_val"],
                p_min=run_stats["defs"]["p_min"],
                p_pixel=run_stats["defs"]["p_pixel"],
                conv_to_int=run_stats["defs"]["conv_to_int"],
            )

    elif run_stats["type_noise"] == "AdditiveGaussian":
        is_random = True

        noise_mdl = ImageAdditiveGaussianNoise(
            array_data.shape,
            probability=run_stats["defs"]["p_img"],
            one_cell_flag=False,
            min_val=run_stats["defs"]["min_val"],
            max_val=run_stats["defs"]["max_val"],
            mu=run_stats["defs"]["mu"],
            sigma=run_stats["defs"]["sigma"],
            scale=np.array([run_stats["defs"]["scale"]]),
            p_pixel=run_stats["defs"]["p_pixel"],
        )

        if array_data_test is not None:
            noise_mdl_test = ImageAdditiveGaussianNoise(
                array_data_test.shape,
                probability=run_stats["defs"]["p_img"],
                one_cell_flag=False,
                min_val=run_stats["defs"]["min_val"],
                max_val=run_stats["defs"]["max_val"],
                mu=run_stats["defs"]["mu"],
                sigma=run_stats["defs"]["sigma"],
                scale=np.array([run_stats["defs"]["scale"]]),
                p_pixel=run_stats["defs"]["p_pixel"],
            )

    else:
        raise ValueError("Noise model type does not exist!")

    if is_random:
        # apply noise model to data
        X_noise, _ = noise_mdl.apply(array_data)

        # reshape to regular format dims: (examples, features)
        X_collapsed = array_data.reshape((num_points, -1))
        X_noise_collapsed = X_noise.reshape((num_points, -1))

        # dataframes with img data
        clean_data_df = pd.DataFrame(X_collapsed, columns=col_names)
        noised_data_df = pd.DataFrame(X_noise_collapsed, columns=col_names)

        # get changes (outlier structs)
        df_changes, cells_changed, tuples_changed = pd_df_diff(
            clean_data_df, noised_data_df
        )

        tuples_changed = tuples_changed.astype(int)
        cells_changed = cells_changed.astype(int)

        noised_data_obj = NoisedDataStruct(
            X_noise_collapsed,  # flatten img dims
            X_collapsed,  # flatten img dims
            y_class_in,
            tuples_changed,
            None,
            None,
        )

        if array_data_test is not None:
            # apply noise model to data (test set)
            X_noise_test, _ = noise_mdl_test.apply(array_data_test)

            # reshape to regular format dims: (examples, features)
            X_collapsed_test = array_data_test.reshape((_test_num_points, -1))
            X_noise_collapsed_test = X_noise_test.reshape((_test_num_points, -1))

            # dataframes with img data
            clean_data_df_test = pd.DataFrame(X_collapsed_test, columns=col_names)
            noised_data_df_test = pd.DataFrame(
                X_noise_collapsed_test, columns=col_names
            )

            # get changes (outlier structs)
            df_changes_test, cells_changed_test, tuples_changed_test = pd_df_diff(
                clean_data_df_test, noised_data_df_test
            )

            tuples_changed_test = tuples_changed_test.astype(int)
            cells_changed_test = cells_changed_test.astype(int)

            noised_data_test_obj = NoisedDataStruct(
                X_noise_collapsed_test,  # flatten img dims
                X_collapsed_test,  # flatten img dims
                y_class_in_test,
                tuples_changed_test,
                None,
                None,
            )

        noise_list_trust = "regular"

    else:
        ## systematic error models

        # reshape to regular format dims: (examples, features)
        X_collapsed = array_data.reshape((num_points, -1))
        X_noise_collapsed = X_noise.reshape((num_points, -1))

        # dataframes with img data
        clean_data_df = pd.DataFrame(X_collapsed, columns=col_names)
        noised_data_df = pd.DataFrame(X_noise_collapsed, columns=col_names)

        # get changes (outlier structs)
        df_changes, cells_changed, tuples_changed = pd_df_diff(
            clean_data_df, noised_data_df
        )

        tuples_changed = noised_data_obj.y_noise  # tuples_changed.astype(int)
        cells_changed = cells_changed.astype(int)

        if array_data_test is not None:
            # reshape to regular format dims: (examples, features)
            X_collapsed_test = array_data_test.reshape((_test_num_points, -1))
            X_noise_collapsed_test = X_noise_test.reshape((_test_num_points, -1))

            # dataframes with img data
            clean_data_df_test = pd.DataFrame(X_collapsed_test, columns=col_names)
            noised_data_df_test = pd.DataFrame(
                X_noise_collapsed_test, columns=col_names
            )

            # get changes (outlier structs)
            df_changes_test, cells_changed_test, tuples_changed_test = pd_df_diff(
                clean_data_df_test, noised_data_df_test
            )

            tuples_changed_test = (
                noised_data_test_obj.y_noise
            )  # tuples_changed.astype(int)
            cells_changed_test = cells_changed_test.astype(int)

        # define what kind of outlier labelling for trusted set
        if (run_stats["noise_list_trusted"] is None) or (
            not run_stats["defs"]["combs_on"]
        ):
            run_stats["noise_list_trusted"] = "regular"

        if run_stats["noise_list_trusted"] not in [
            "label_combs",
            "regular",
        ]:
            raise ValueError(
                "Option for 'noise_list_trusted' unclear. Either use: 'regular' or 'label_combs' "
            )
        # "label_combs": examples with combinations of errors are labelled uniquely (e.g. use in stratisfied sampling).
        # "regular": usual y_noise_lists, does not guarantee or provide labels on noise combinations.

        noise_list_trust = run_stats["noise_list_trusted"]

    # struct for entire dataset (no splitting)
    if test_set_given:
        _full_dataset = dict()
        _full_dataset["df_data"] = pd.concat(
            [clean_data_df, clean_data_df_test], axis=0, ignore_index=True
        )
        _full_dataset["df_noise_data"] = pd.concat(
            [noised_data_df, noised_data_df_test], axis=0, ignore_index=True
        )
        _full_dataset["cells_changed_oh"] = np.concatenate(
            (cells_changed, cells_changed_test), axis=0
        )
        _full_dataset["tuples_changed_oh"] = np.concatenate(
            (tuples_changed, tuples_changed_test), axis=0
        )
        if noised_data_obj.y_class is not None:
            _full_dataset["y_class"] = np.concatenate(
                [noised_data_obj.y_class, noised_data_test_obj.y_class]
            )
        else:
            _full_dataset["y_class"] = None
        if noised_data_obj.y_noise_lists is not None:
            _full_dataset["y_noise_lists"] = _combine_y_lists(
                noised_data_obj.y_noise_lists,
                noised_data_test_obj.y_noise_lists,
                offset=num_points,
            )
        else:
            _full_dataset["y_noise_lists"] = None

    else:
        _full_dataset = dict()
        _full_dataset["df_data"] = clean_data_df
        _full_dataset["df_noise_data"] = noised_data_df
        _full_dataset["cells_changed_oh"] = cells_changed
        _full_dataset["tuples_changed_oh"] = tuples_changed
        _full_dataset["y_class"] = noised_data_obj.y_class
        _full_dataset["y_noise_lists"] = noised_data_obj.y_noise_lists

    ## get dataset splits (train, validation, test)
    train_idxs, validation_idxs, test_idxs = create_data_splits(
        run_stats, clean_data_df, test_set_given
    )

    if test_set_given:
        test_set_in = dict()
        test_set_in["df_data"] = clean_data_df_test
        test_set_in["df_noise_data"] = noised_data_df_test
        test_set_in["cells_changed_oh"] = cells_changed_test
        test_set_in["tuples_changed_oh"] = tuples_changed_test
        test_set_in["y_class"] = noised_data_test_obj.y_class
        test_set_in["y_noise_lists"] = noised_data_test_obj.y_noise_lists
        test_set_in["y_noise_lists_combs"] = noised_data_test_obj.y_noise_lists_combs
    else:
        test_set_in = None

    ret = _enforce_data_split(
        clean_data_df,
        train_idxs,
        validation_idxs,
        test_idxs,
        noised_data_df,
        cells_changed,
        tuples_changed,
        noised_data_obj.y_class,
        noised_data_obj.y_noise_lists,
        noised_data_obj.y_noise_lists_combs,
        test_set_in,
    )
    _train_dataset, _valid_dataset, _test_dataset = ret

    ## get trusted set (subset of training dataset, for semi-supervised labelling)
    if not run_stats["trusted_set"]["y_class_on"]:
        train_y_class_in = None
    else:
        # use underlying clean labels for trust-set build
        train_y_class_in = _train_dataset["y_class"]

    if not run_stats["trusted_set"]["y_noise_lists_on"]:
        train_y_noise_lists_in = None
    else:
        if noise_list_trust == "regular":
            train_y_noise_lists_in = _train_dataset["y_noise_lists"]
        elif noise_list_trust == "label_combs":
            train_y_noise_lists_in = _train_dataset["y_noise_lists_combs"]
            # NOTE: y_noise_lists_combs is not being saved, for trust set build below mainly.

    trust_set_idxs, trust_set_dicts = create_trust_set_split(
        run_stats,
        _train_dataset["df_data"],
        label_type=run_stats["trusted_set"]["use_labels"],
        y_class=train_y_class_in,
        y_noise=_train_dataset["tuples_changed_oh"],
        y_noise_lists=train_y_noise_lists_in,
    )

    if trust_set_dicts:
        # NOTE: exists -> np.array or list for run_stats["trusted_set"] input: fraction / fixed samples
        _ts_input = trust_set_dicts

        if run_stats["trusted_set"]["mc_mode"] == "fixed_number":
            ts_type = "samples"

        elif (run_stats["trusted_set"]["mc_mode"] == "stratisfied_v2") or (
            not run_stats["trusted_set"]["min_coverage"]
        ):
            ts_type = "percent"
    else:
        _ts_input = trust_set_idxs
        ts_type = None

    # save to folders
    _save_datasets(
        path_saving,
        _train_dataset,
        _valid_dataset,
        _test_dataset,
        _full_dataset,
        _ts_input,
        trust_set_type=ts_type,
    )

    cols_info = {
        "cat_cols_names": [],
        "num_cols_names": col_names,
        "dataset_type": "image",
        "image_defs": image_defs,
    }

    with open(path_saving + "cols_info.json", "w") as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    with open(path_saving + "noising_info.json", "w") as outfile:
        json.dump(run_stats, outfile, indent=4, sort_keys=True)
