from scipy.io import loadmat
import numpy as np
import pandas as pd
import json

import sys
import os

sys.path.append(os.getcwd() + "/..")

from dataset_prep_utils_semisup import dirty_semi_supervised_image


# original file name path
data_filename = "../../../data/FreyFaces/frey_rawface.mat"


## get data, and reshape data
img_x, img_y = 28, 20  # image size
data_arr = loadmat(data_filename, squeeze_me=True, struct_as_record=False)
data_arr = data_arr["ff"].T.reshape((-1, img_x, img_y))  # numpy array
image_size_ff = [28, 20]

## save folder path
save_folder = "../../../data/FreyFaces/testing_runs/"

## type of noise to be injected
run_noise_option = "simple_systematic"  # salt_n_pepper ; simple_systematic

## noise definitions
if run_noise_option == "simple_systematic":

    # run_stats = {
    #     "name": "bogus_run",
    #     "train_size": 0.8,
    #     "valid_size": 0.1,
    #     "test_size": 0.1,
    #     "trusted_set": {
    #         "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
    #         "min_coverage": True,  # True
    #         "mc_mode": "stratisfied_v2",  # "fixed_number" "stratisfied_v2", fixed ?
    #         "samples_fixed": 10,
    #         "frac_trusted": 0.10,  # ignored if samples fixed.
    #         "y_class_on": True,
    #         "y_noise_lists_on": True,
    #     },
    #     "type_noise": "SystematicSimpleShapes",
    #     "defs": {
    #         "p_img": 0.25,  # corruption probability
    #         "min_val": 0,
    #         "max_val": 256,  # check this, and if freyfaces needs standardizing beforehand.
    #         "p_min": 0.5,
    #         "pixel_val_fixed": None,  # otherwise fix a value value; None or value (e.g. 150)
    #         "number_blocks": 4,  # 1 ; 2; 4
    #         "rand_blocks": True,
    #         "side_len": 4,  # 4 ; 11
    #         "std_shift": (10, 10),  # (10,10) ; (5, 6)
    #         "use_other_patterns": False,
    #         "random_state": None,  # seed number
    #         "combs_on": False,  # True / False
    #     },
    #     "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
    # }

    run_stats = {
        "name": "bogus_run_balanced",
        "train_size": 0.8,
        "valid_size": 0.1,
        "test_size": 0.1,
        "trusted_set": {
            "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
            "min_coverage": True,  # True
            "mc_mode": "fixed_number",  # "fixed_number" "stratisfied_v2", fixed ?
            "samples_fixed": 10,
            "frac_trusted": 0.10,  # ignored if samples fixed.
            "y_class_on": True,
            "y_noise_lists_on": True,
        },
        "type_noise": "SystematicSimpleShapes",
        "defs": {
            "p_img": 0.25,  # corruption probability
            "min_val": 0,
            "max_val": 256,  # check this, and if freyfaces needs standardizing beforehand.
            "p_min": 0.5,
            "pixel_val_fixed": None,  # otherwise fix a value value; None or value (e.g. 150)
            "number_blocks": 4,  # 1 ; 2; 4
            "rand_blocks": True,
            "side_len": 4,  # 4 ; 11
            "std_shift": (10, 10),  # (10,10) ; (5, 6)
            "use_other_patterns": False,
            "random_state": None,  # seed number
            "combs_on": False,  # True / False
        },
        "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
    }

    # NOTE:

    # img_prob_noise=run_stats["defs"]["p_img"],
    # min_val=run_stats["defs"]["min_val"],
    # max_val=run_stats["defs"]["max_val"],
    # prob_min=run_stats["defs"]["p_min"],
    # pixel_val_fixed=run_stats["defs"]["pixel_val_fixed"],
    # number_blocks=run_stats["defs"]["number_blocks"],
    # rand_blocks=run_stats["defs"]["rand_blocks"],
    # side_len=run_stats["defs"]["side_len"],
    # std_shift=run_stats["defs"]["std_shift"],
    # use_other_patterns=run_stats["defs"]["use_other_patterns"],
    # random_state=run_stats["defs"]["random_state"],
    # combs_on=run_stats["defs"]["combs_on"],

    # ["noise_list_trusted"] ;

    # ["trusted_set"]["y_class_on"] ; ["trusted_set"]["y_noise_lists_on"] ; ["trusted_set"]["use_labels"] ;

    # ["trusted_set"]["use_labels"] := "clean_class_only" / "dirty_classes_only" / "joint_classes"

elif run_noise_option == "salt_n_pepper":

    run_stats = {
        "name": "bogus_run",
        "train_size": 0.8,
        "valid_size": 0.1,
        "test_size": 0.1,
        "trusted_set": {
            "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
            "min_coverage": True,  # True
            "mc_mode": "stratisfied_v2",  # "fixed_number" "stratisfied_v2", fixed ?
            "samples_fixed": 10,
            "frac_trusted": 0.10,  # ignored if samples fixed.
            "y_class_on": True,
            "y_noise_lists_on": True,
        },
        "type_noise": "SaltnPepper",
        "defs": {
            "p_img": 0.25,  # image corruption probability
            "min_val": 0,
            "max_val": 256,  # check this, and if freyfaces needs standardizing beforehand.
            "p_min": 0.5,
            "p_pixel": 0.2,  # pixel corruption probability
            "conv_to_int": False,
        },
    }

    # NOTE:

    # probability=run_stats["defs"]["p_img"],
    # one_cell_flag=False,
    # min_val=run_stats["defs"]["min_val"],
    # max_val=run_stats["defs"]["max_val"],
    # p_min=run_stats["defs"]["p_min"],
    # p_pixel=run_stats["defs"]["p_pixel"],
    # conv_to_int=run_stats["defs"]["conv_to_int"],

## apply noise model
dirty_semi_supervised_image(
    data_arr, run_stats, save_folder, image_size_ff, y_class_in=None
)
