import pandas as pd
import numpy as np
import os, errno
from dataset_prep_utils_semisup import dirty_semi_supervised_synthethic


# save_folder = "../../data/SimpleShapesWithStrips/"

# run_stats = {
#     "name": "bogus_run",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,  # True
#         "mc_mode": None,  # "stratisfied_v2"
#         "frac_trusted": 0.20,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "ShapesWithStrips",
#         "defs": {
#             "n_samples": 600,
#             "corrupt_prob": 0.20,  # 0.10
#             "random_state": None,  # seed number
#             "combs_on": True,
#         },
#         "noise_list_trusted": "label_combs",  # "regular"
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


## Example Simple Shapes -- NO COMBINATIONS

# save_folder = "../../data/SimpleShapesWithStrips/"

# run_stats = {
#     "name": "bogus_run_v2",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,  # True
#         "mc_mode": "stratisfied_v2",  # "stratisfied_v2"
#         "frac_trusted": 0.20,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "ShapesWithStrips",
#         "defs": {
#             "n_samples": 1000,
#             "corrupt_prob": 0.15,  # 0.10, 0.40
#             "random_state": None,  # seed number
#             "combs_on": False,  # False
#         },
#         "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


## Example Simple Shapes -- WITH COMBINATIONS

# save_folder = "../../data/SimpleShapesWithStrips/"

# run_stats = {
#     "name": "bogus_run",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,  # True
#         "mc_mode": "stratisfied_v2",  # "stratisfied_v2"
#         "frac_trusted": 0.10,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "ShapesWithStrips",
#         "defs": {
#             "n_samples": 1000,
#             "corrupt_prob": 0.30,  # 0.10, 0.40
#             "random_state": None,  # seed number
#             "combs_on": True,  # False
#         },
#         "noise_list_trusted": "label_combs",  # "regular" or "label_combs" # type of labelling for trusted set, dataset dependent
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


############

# Example Simple Shapes -- NO COMBINATIONS (EASY MODE)

save_folder = "../../data/SimpleShapesWithStrips/"

run_stats = {
    "name": "bogus_run",
    "train_size": 0.8,
    "valid_size": 0.1,
    "test_size": 0.1,
    "trusted_set": {
        "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
        "min_coverage": True,  # True
        "mc_mode": "fixed_number",  # "stratisfied_v2", fixed ?
        "samples_fixed": 10,
        "frac_trusted": 0.30,  # ignored if samples fixed.
        "y_class_on": True,
        "y_noise_lists_on": True,
    },
    "synth_data": {
        "type": "ShapesWithStrips",
        "defs": {
            "n_samples": 1000,
            "corrupt_prob": 0.15,  # 0.10, 0.40
            "random_state": None,  # seed number
            "combs_on": False,  # False
        },
        "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
    },
}

dirty_semi_supervised_synthethic(run_stats, save_folder)
