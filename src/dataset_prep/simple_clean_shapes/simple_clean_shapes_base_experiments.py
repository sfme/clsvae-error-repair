import pandas as pd
import numpy as np
import json

import sys
import os

sys.path.append(os.getcwd() + "/..")

from dataset_prep_utils_semisup import dirty_semi_supervised_synthethic

n_datapoints = 5000


## save folder path
# base_experiments/[balanced; stratisfied]/[dataset]/[corrupt_level_[number]p[number]]/[trustedset_level_[number]p[number]]]
save_folder = f"../../../data/base_experiments/balanced/simple_shapes/noisy_lines/"

## noise levels
noise_level = 35  # percentage of corruption -- 15; 25 x ; 35 ; 45
save_folder = save_folder + f"corrupt_level_{noise_level}_percent/"

## total number of classes (n_classes)
n_classes = 8  # 4: clean class ; 4: dirty classes NOTE: update if needed !!

## trust set sweep values
# size of overall dataset and trusted set size sweeps
per_class_points = np.array([5, 10, 25, 50, 100])
# NOTE: (up to 16% dataset)
trust_set_percent = (per_class_points * n_classes) / n_datapoints * 100
# TODO: maybe save number of samples instead of % of trusted set?

# loop through sweeps (trusted set levels)
for ts_idx, ts_level in enumerate(trust_set_percent):

    save_folder_temp = save_folder + f"trustset_level_{ts_level:.2f}_percent/"
    name_run = "default"

    print(save_folder_temp + name_run + " \n\n\n")

    ## noise definitions
    run_stats = {
        "name": name_run,
        "train_size": 0.8,
        "valid_size": 0.1,
        "test_size": 0.1,
        "trusted_set": {
            "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
            "min_coverage": True,  # True
            "mc_mode": "fixed_number",  # "stratisfied_v2", fixed ?
            "samples_fixed": int(per_class_points[ts_idx]),
            "frac_trusted": None,  # ignored if samples fixed.
            "y_class_on": True,
            "y_noise_lists_on": True,
        },
        "synth_data": {
            "type": "ShapesWithStrips",
            "defs": {
                "n_samples": n_datapoints,
                "corrupt_prob": noise_level / 100.0,  # 0.10, 0.40
                "random_state": None,  # seed number
                "combs_on": False,  # False
            },
            "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
        },
    }

    ## apply noise model
    dirty_semi_supervised_synthethic(run_stats, save_folder_temp)
