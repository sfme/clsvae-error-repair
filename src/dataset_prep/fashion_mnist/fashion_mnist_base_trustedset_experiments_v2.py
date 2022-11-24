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
n_test_datapoints = test_array.shape[0]

# size of entire dataset
n_datapoints = n_train_datapoints + n_test_datapoints

#### VERSION I

## type of noise to be injected
run_noise_option = "simple_systematic"  # salt_n_pepper ; simple_systematic

## total number of classes (n_classes)
n_classes = 10 + 8  # 10: clean class ; 8: dirty classes

## save folder path
# base_experiments/[balanced; stratisfied]/[dataset]/[corrupt_level_[number]]/
save_folder = (
    f"../../../data/base_experiments_v2/balanced/fashion_mnist/{run_noise_option}/"
)

## noise levels
noise_level_percentages = [15, 25, 35, 45]

## trust set sweep values
per_class_points = np.array([5, 10, 25, 50], dtype=int)

trust_set_total_points = per_class_points * n_classes
trust_set_percent = trust_set_total_points / n_datapoints * 100


# run / new random seed
# name_run = "run_1"  # NOTE: 5 runs. # 1,2,3,4,5
for cur_run in range(5):
    name_run = f"run_{cur_run+1}"  # number of runs

    ## noise levels
    # noise_level = 25  # percentage of corruption -- 15; 25 ; 35 ; 45
    for noise_level_cur in noise_level_percentages:

        save_folder_cur = save_folder + f"corrupt_level_{noise_level_cur}_percent/"

        print(save_folder_cur + name_run + " \n\n\n")

        ## noise definitions
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
                    "samples_fixed": per_class_points.tolist(),
                    "frac_trusted": None,  # ignored if samples fixed.
                    "y_class_on": True,
                    "y_noise_lists_on": True,
                },
                "type_noise": "SystematicSimpleShapes",
                "defs": {
                    "p_img": noise_level_cur / 100.0,  # corruption probability
                    "min_val": 0,
                    "max_val": 1,
                    "p_min": 0.5,
                    "pixel_val_fixed": None,  # None  # otherwise fix a value value; None or value (e.g. 150)
                    "number_blocks": 4,  # 1 ; 2; 4
                    "rand_blocks": True,
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
                    "samples_fixed": per_class_points.tolist(),
                    "frac_trusted": None,  # ignored if samples fixed.
                    "y_class_on": True,
                    "y_noise_lists_on": True,
                },
                "type_noise": "SaltnPepper",
                "defs": {
                    "p_img": noise_level_cur / 100.0,  # corruption probability
                    "min_val": 0,
                    "max_val": 1,
                    "p_min": 0.5,
                    "p_pixel": 0.30,  # pixel corruption probability
                    "conv_to_int": False,
                },
            }

        ## update run_stats with some dataset definitions
        run_stats["trusted_set"]["n_classes"] = n_classes
        run_stats["trusted_set"]["total_points"] = trust_set_total_points.tolist()
        run_stats["trusted_set"]["percentages"] = trust_set_percent.tolist()
        run_stats["trusted_set"]["dataset_size"] = n_datapoints

        ## apply noise model
        dirty_semi_supervised_image(
            train_array,
            run_stats,
            save_folder_cur,
            image_size,
            y_class_in=train_labels,
            array_data_test=test_array,
            y_class_in_test=test_labels,
        )
