import pandas as pd
import numpy as np
import os, errno
from dataset_prep_utils_semisup import dirty_semi_supervised_synthethic


############
## SIMPLE SHAPES WITH STRIPS -- TYPE DATASET
## Using Simple Example (NO COMBINATIONS -- EASIER)


gen_save_folder = "../../data/7_DEC_2020/"

run_mode = "easy_scenario"  # 'noise_levels' or 'trusted_sets' or 'easy_scenario'

if run_mode == "trusted_sets":
    ### SWEEP TRUSTED SET SIZE

    n_datapoints = 5000

    trust_set_percent = np.array([2, 5, 8, 10, 15, 20, 30])
    trust_set_portion = trust_set_percent / 100.0

    corrup_percent = 25.0

    save_folder = gen_save_folder + "trusted_set_sizes/" + "fixed_number/"

    ## Using Fixed Number of Samples per Class

    n_classes = 8  # includes clean and dirty classes
    per_class_points = np.ceil((trust_set_portion * n_datapoints) / n_classes).astype(
        int
    )

    jj = 0
    for ts_size in trust_set_percent:

        name_dataset = f"ts_size_{ts_size}_percent"

        print(save_folder + name_dataset + " \n\n\n")

        run_stats = {
            "name": name_dataset,
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "trusted_set": {
                "use_labels": "joint_classes",  # "joint_classes"; no_labels; joint_classes; dirty_classes_only
                "min_coverage": True,  # True
                "mc_mode": "fixed_number",  # "stratisfied_v2", fixed ?
                "samples_fixed": int(per_class_points[jj]),
                "frac_trusted": trust_set_portion[
                    jj
                ],  # NOTE: ignored if fixed samples used, but will save noising_info.json
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "synth_data": {
                "type": "ShapesWithStrips",
                "defs": {
                    "n_samples": n_datapoints,
                    "corrupt_prob": corrup_percent / 100.0,
                    "random_state": None,  # seed number
                    "combs_on": False,  # False
                },
                "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
            },
        }
        dirty_semi_supervised_synthethic(run_stats, save_folder)
        jj += 1

    ## Using Stratisfied Sampling (includes more clean labels, and less dirty ones -- naturally)

    save_folder = gen_save_folder + "trusted_set_sizes/" + "stratisfied/"

    jj = 0
    for ts_size in trust_set_percent:

        name_dataset = f"ts_size_{ts_size}_percent"

        print(save_folder + name_dataset + " \n\n\n")

        run_stats = {
            "name": name_dataset,
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "trusted_set": {
                "use_labels": "joint_classes",
                "min_coverage": True,
                "mc_mode": "stratisfied_v2",
                "frac_trusted": trust_set_portion[jj],
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "synth_data": {
                "type": "ShapesWithStrips",
                "defs": {
                    "n_samples": n_datapoints,
                    "corrupt_prob": corrup_percent / 100.0,
                    "random_state": None,  # seed number
                    "combs_on": False,  # False
                },
                "noise_list_trusted": "regular",  # type of labelling for trusted set, dataset dependent
            },
        }
        dirty_semi_supervised_synthethic(run_stats, save_folder)
        jj += 1


elif run_mode == "noise_levels":
    ### SWEEP NOISE LEVELS

    n_datapoints = 5000

    # TODO: saving via new folder name for different (trust set size) or (sampling type)?

    save_folder = gen_save_folder + "noise_levels/"

    corruption_percent = np.array([5, 10, 15, 25, 35])
    corruption_portion = corruption_percent / 100.0

    trust_set_portion = 5 / 100.0  # or 10 percent ? (NOTE: can try after another level)

    ## Using Fixed Number of Samples per Class

    n_classes = 8  # includes clean and dirty classes
    n_fixed_samples = int(np.ceil((trust_set_portion * n_datapoints) / n_classes))

    jj = 0
    for noise_level in corruption_percent:

        name_dataset = f"corrup_size_{noise_level}_percent"

        print(save_folder + name_dataset + " \n\n\n")

        run_stats = {
            "name": name_dataset,
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "trusted_set": {
                "use_labels": "joint_classes",
                "min_coverage": True,
                "mc_mode": "fixed_number",
                "samples_fixed": n_fixed_samples,
                "frac_trusted": trust_set_portion,  # NOTE: ignored if samples_fixed used!
                "y_class_on": True,
                "y_noise_lists_on": True,
            },
            "synth_data": {
                "type": "ShapesWithStrips",
                "defs": {
                    "n_samples": n_datapoints,
                    "corrupt_prob": corruption_portion[jj],
                    "random_state": None,  # seed number
                    "combs_on": False,  # False
                },
                "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
            },
        }
        dirty_semi_supervised_synthethic(run_stats, save_folder)
        jj += 1


### Upper-Bound Performance (bigger dataset, and larger trusted set)

elif run_mode == "easy_scenario":

    n_datapoints = 5000

    save_folder = gen_save_folder + "easy_scenario/"

    corruption_portion = 0.40  # 0.20; 0.10; 0.40

    trust_set_portion = 0.80  # 0.80

    # n_classes = 8  # includes clean and dirty classes
    # n_fixed_samples = int(np.ceil((trust_set_portion * n_datapoints) / n_classes))

    name_dataset = f"noise_size_{corruption_portion}_ts_size_{trust_set_portion}"

    print(save_folder + name_dataset + " \n\n\n")

    run_stats = {
        "name": name_dataset,
        "train_size": 0.8,
        "valid_size": 0.1,
        "test_size": 0.1,
        "trusted_set": {
            "use_labels": "joint_classes",
            "min_coverage": True,
            "mc_mode": "stratisfied_v2",  # stratisfied_v2 fixed_number
            "samples_fixed": None,  # n_fixed_samples
            "frac_trusted": trust_set_portion,  # NOTE: ignored if samples_fixed used!
            "y_class_on": True,
            "y_noise_lists_on": True,
        },
        "synth_data": {
            "type": "ShapesWithStrips",
            "defs": {
                "n_samples": n_datapoints,
                "corrupt_prob": corruption_portion,
                "random_state": None,  # seed number
                "combs_on": False,  # False
            },
            "noise_list_trusted": "regular",  # "regular" # type of labelling for trusted set, dataset dependent
        },
    }
    dirty_semi_supervised_synthethic(run_stats, save_folder)

else:
    raise ValueError("Option for creating dataset does not exist !!")
