import pandas as pd
import numpy as np
import json

import sys
import os

sys.path.append(os.getcwd() + "/..")

from dataset_prep_utils_semisup import dirty_semi_supervised_synthethic

## size of dataset
n_datapoints = 5000

## total number of classes (n_classes)
n_classes = 8  # 4: clean class ; 4: dirty classes NOTE: update if needed !!

## save folder path
# base_experiments/[balanced; stratisfied]/[dataset]/[corrupt_level_[number]]/
save_folder = f"../../../data/base_experiments_v2/balanced/simple_shapes/noisy_lines/"

noise_level_percentages = [15, 25, 35, 45]

## trust set sweep values
# size of overall dataset and trusted set size sweeps
per_class_points = np.array([5, 10, 25, 50], dtype=int)
# NOTE: (up to 16% dataset)

trust_set_total_points = per_class_points * n_classes
trust_set_percent = trust_set_total_points / n_datapoints * 100

# run / new random seed
# name_run = "run_1"  # NOTE: 5 runs. # 1,2,3,4,5
for cur_run in range(5):
    name_run = f"run_{cur_run+1}"  # number of runs

    ## noise levels
    for noise_level_cur in noise_level_percentages:

        save_folder_cur = save_folder + f"corrupt_level_{noise_level_cur}_percent/"

        print(save_folder_cur + name_run + " \n\n\n")

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
                "samples_fixed": per_class_points.tolist(),
                "frac_trusted": None,  # ignored if samples fixed.
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "synth_data": {
                "type": "ShapesWithStrips",
                "defs": {
                    "n_samples": n_datapoints,
                    "corrupt_prob": noise_level_cur / 100.0,  # 0.10, 0.40
                    "random_state": None,  # seed number
                    "combs_on": False,  # False
                },
                "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
            },
        }

        ## update run_stats with some dataset definitions
        run_stats["trusted_set"]["n_classes"] = n_classes
        run_stats["trusted_set"]["total_points"] = trust_set_total_points.tolist()
        run_stats["trusted_set"]["percentages"] = trust_set_percent.tolist()
        run_stats["trusted_set"]["dataset_size"] = n_datapoints

        dirty_semi_supervised_synthethic(run_stats, save_folder_cur)
