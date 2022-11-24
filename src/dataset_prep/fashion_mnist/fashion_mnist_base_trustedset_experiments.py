import numpy as np
import pandas as pd
import json

import sys
import os

sys.path.append(os.getcwd() + "/..")

from dataset_prep_utils_semisup import dirty_semi_supervised_image


# original file name path
data_filename = "../../../data/FashionMNIST/"
path_orig_data = data_filename + "original_raw/"  # needs mkdir

# load original (clean) data
with open(path_orig_data + "train_images.npy", "rb") as f:
    train_array = np.load(f)

with open(path_orig_data + "train_labels.npy", "rb") as f:
    train_labels = np.load(f)

with open(path_orig_data + "test_images.npy", "rb") as f:
    test_array = np.load(f)

with open(path_orig_data + "test_labels.npy", "rb") as f:
    test_labels = np.load(f)

# image size
image_size = [28, 28]

n_train_datapoints = train_array.shape[0]

#### VERSION I

## type of noise to be injected
run_noise_option = "simple_systematic"  # salt_n_pepper ; simple_systematic

## save folder path
# base_experiments/[balanced; stratisfied]/[dataset]/[corrupt_level_[number]p[number]]/[trustedset_level_[number]p[number]]]
save_folder = (
    f"../../../data/base_experiments/balanced/fashion_mnist/{run_noise_option}/"
)

## noise levels
noise_level = 25  # percentage of corruption -- 15; 25 x ; 35 x ; 45
save_folder = save_folder + f"corrupt_level_{noise_level}_percent/"

## total number of classes (n_classes)
n_classes = 10 + 8  # 10: clean class ; 8: dirty classes

## trust set sweep values
# size of overall dataset and trusted set size sweeps
per_class_points = np.array([5, 10, 25, 50, 100])
trust_set_percent = (per_class_points * n_classes) / n_train_datapoints * 100


# loop through sweeps (trusted set levels)
for ts_idx, ts_level in enumerate(trust_set_percent):

    save_folder_temp = save_folder + f"trustset_level_{ts_level:.2f}_percent/"
    name_run = "default"

    print(save_folder_temp + name_run + " \n\n\n")

    if run_noise_option == "simple_systematic":

        run_stats = {
            "name": name_run,
            "train_size": 0.9,
            "valid_size": 0.1,
            "test_size": None,
            "trusted_set": {
                "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
                "min_coverage": True,  # True
                "mc_mode": "fixed_number",  # "fixed_number" "stratisfied_v2", fixed ?
                "samples_fixed": int(per_class_points[ts_idx]),
                "frac_trusted": None,  # ignored if samples fixed.
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "type_noise": "SystematicSimpleShapes",
            "defs": {
                "p_img": noise_level / 100.0,  # corruption probability
                "min_val": 0,
                "max_val": 1,
                "p_min": 0.5,
                "pixel_val_fixed": None,  # None  # otherwise fix a value value; None or value (e.g. 150)
                "number_blocks": 4,  # 1 ; 2; 4
                "rand_blocks": False,
                "side_len": 6,  # 4 ; 11
                "std_shift": (10, 10),  # (10,10) ; (5, 6)
                "use_other_patterns": True,
                "random_state": None,  # seed number
                "combs_on": False,  # True / False
            },
            "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
        }

    elif run_noise_option == "salt_n_pepper":

        run_stats = {
            "name": name_run,
            "train_size": 0.9,
            "valid_size": 0.1,
            "test_size": None,
            "trusted_set": {
                "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
                "min_coverage": True,  # True
                "mc_mode": "fixed_number",  # "fixed_number" "stratisfied_v2", fixed ?
                "samples_fixed": int(per_class_points[ts_idx]),
                "frac_trusted": None,  # ignored if samples fixed.
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "type_noise": "SaltnPepper",
            "defs": {
                "p_img": noise_level / 100.0,  # corruption probability
                "min_val": 0,
                "max_val": 1,
                "p_min": 0.5,
                "p_pixel": 0.30,  # pixel corruption probability
                "conv_to_int": False,
            },
        }

    ## apply noise model
    dirty_semi_supervised_image(
        train_array,
        run_stats,
        save_folder_temp,
        image_size,
        y_class_in=train_labels,
        array_data_test=test_array,
        y_class_in_test=test_labels,
    )
