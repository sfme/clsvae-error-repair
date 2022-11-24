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

n_datapoints = data_arr.shape[0]

#### VERSION I

## type of noise to be injected
run_noise_option = "simple_systematic_shapes"  # salt_n_pepper ; simple_systematic

## save folder path
# base_experiments/[balanced; stratisfied]/[dataset]/[corrupt_level_[number]p[number]]/[trustedset_level_[number]p[number]]]
save_folder = f"../../../data/base_experiments/balanced/frey_faces/{run_noise_option}/"

## noise levels
noise_level = 25  # percentage of corruption -- 15; 25 x ; 35 x ; 45
save_folder = save_folder + f"corrupt_level_{noise_level}_percent/"

## total number of classes (n_classes)
n_classes = 5  # 1: clean class ; 4: dirty classes NOTE: update if needed !!

## trust set sweep values
# size of overall dataset and trusted set size sweeps
per_class_points = np.array([5, 10, 25, 50])
# 25 samples: np.array([5, 10, 25, 50])
# 35 samples and above: np.array([5, 10, 25, 50, 100])
trust_set_percent = (per_class_points * n_classes) / n_datapoints * 100


# loop through sweeps (trusted set levels)
for ts_idx, ts_level in enumerate(trust_set_percent):

    save_folder_temp = save_folder + f"trustset_level_{ts_level:.2f}_percent/"
    name_run = "default"

    print(save_folder_temp + name_run + " \n\n\n")

    ## noise definitions
    if run_noise_option == "simple_systematic_shapes":

        run_stats = {
            "name": name_run,
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "trusted_set": {
                "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
                "min_coverage": True,  # True
                "mc_mode": "fixed_number",  # "fixed_number" "stratisfied_v2", fixed ?
                "samples_fixed": int(per_class_points[ts_idx]),
                "frac_trusted": None,  # 0.10 ignored if samples fixed.
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "type_noise": "SystematicSimpleShapes",
            "defs": {
                "p_img": noise_level / 100.0,  # corruption probability
                "min_val": 0,
                "max_val": 256,  # check this, and if freyfaces needs standardizing beforehand.
                "p_min": 0.5,
                "pixel_val_fixed": None,  # otherwise fix a value value; None or value (e.g. 150)
                "number_blocks": 4,  # 1 ; 2; 4
                "rand_blocks": True,
                "side_len": 6,  # 4 ; 11
                "std_shift": (10, 10),  # (10,10) ; (5, 6)
                "use_other_patterns": False,
                "random_state": None,  # seed number
                "combs_on": False,  # True / False
            },
            "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
        }

    elif run_noise_option == "salt_n_pepper":

        run_stats = {
            "name": name_run,
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
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
                "p_img": noise_level / 100.0,  # image corruption probability
                "min_val": 0,
                "max_val": 256,  # check this, and if freyfaces needs standardizing beforehand.
                "p_min": 0.5,
                "p_pixel": 0.30,  # pixel corruption probability
                "conv_to_int": False,
            },
        }

    ## apply noise model
    dirty_semi_supervised_image(
        data_arr, run_stats, save_folder_temp, image_size_ff, y_class_in=None
    )
