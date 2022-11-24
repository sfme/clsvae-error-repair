import pandas as pd
import numpy as np
import os, errno
from dataset_prep_utils_semisup import dirty_semi_supervised_synthethic


## Testing Synthetic Noisy GMM cases

# save_folder = '../../data/GMM_Synth/'

# run_stats = {
#     'name': 'simple_trial',
#     'train_size': 0.8,
#     'valid_size':0.1,
#     'test_size':0.1,
#     'trusted_set':{'use_labels': 'joint_classes', # 'no_labels'
#                    'mc_mode': 'fixed_number', # 'stratisfied_v2'
#                    'samples_fixed': 2,
#                    'min_coverage':True,
#                    'frac_trusted': 0.05,
#                    'y_class_on':True,
#                    'y_noise_lists_on':True},
#     'synth_data':{'type': 'GaussianClusters',
#                   'defs':{
#                           'n_samples': 2000,
#                           'n_clusters': 2,
#                           'corrupt_prob': 0.25,
#                           'n_features': 2,
#                           'scale_density': 'same',
#                           'size_cluster': 'same',
#                           'dist': 0.25,
#                           'random_state': None, # seed number
#                           'noise_type': 'type_1' # 'type_2'
#                           }
#                  }
# }


# run_stats = {
#     'name': 'simple_trial',
#     'train_size': 0.8,
#     'valid_size':0.1,
#     'test_size':0.1,
#     'trusted_set':{'use_labels': 'no_labels',
#                    'min_coverage':False,
#                    'frac_trusted': 0.05,
#                    'y_class_on':True,
#                    'y_noise_lists_on':True,
#                    },
#     'synth_data':{'type': 'GaussianClusters',
#                   'defs':{
#                           'n_samples': 600,
#                           'n_clusters': 5,
#                           'corrupt_prob': 0.3,
#                           'n_features': 2,
#                           'scale_density': 'same',
#                           'size_cluster': 'same',
#                           'dist': 0.3, # 0.25
#                           'random_state': None, # seed number
#                           'noise_type': 'type_2', # 'type_1'
#                           'noise_type_defs':{'n_groups':1, 'err_min':3, 'err_max':4}  # type_2
#                           }
#                  }
# }

# save_folder = '../../data/GMM_Synth_2/'

# run_stats = {
#     'name': 'simple_trial',
#     'train_size': 0.8,
#     'valid_size':0.1,
#     'test_size':0.1,
#     'trusted_set':{'use_labels': 'no_labels',
#                    'min_coverage':False,
#                    'frac_trusted': 0.05,
#                    'y_class_on':True,
#                    'y_noise_lists_on':True,
#                    },
#     'synth_data':{'type': 'GaussianClusters',
#                   'defs':{
#                           'n_samples': 5000,
#                           'n_clusters': 3,
#                           'corrupt_prob': 0.45,
#                           'n_features': 2,
#                           'scale_density': 'same',
#                           'size_cluster': 'same',
#                           'dist': 0.5, # 0.25
#                           'random_state': None, # seed number
#                           'noise_type': 'type_2', # 'type_1'
#                           'noise_type_defs':{'n_groups':1, 'err_min':25, 'err_max':26}  # type_2
#                           }
#                  }
# }


# save_folder = "../../data/GMM_Synth_type1/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",
#         "min_coverage": False,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 5000,
#             "n_clusters": 3,
#             "corrupt_prob": 0.45,
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.5,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_1",  # 'type_1'
#         },
#     },
# }


# save_folder = "../../data/GMM_Synth_3/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",
#         "min_coverage": False,
#         "frac_trusted": 0.02,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 3,
#             "corrupt_prob": 0.15,
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.5,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 1, "err_min": 10, "err_max": 15},  # type_2
#         },
#     },
# }

# save_folder = "../../data/GMM_Synth_4/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",
#         "min_coverage": False,
#         "frac_trusted": 0.02,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 500,
#             "n_clusters": 6,
#             "corrupt_prob": 0.10,
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.5,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 1, "err_min": 15, "err_max": 20},  # type_2
#         },
#     },
# }


# save_folder = "../../data/GMM_Synth_5_high_labels/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",
#         "min_coverage": False,
#         "frac_trusted": 0.35,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 500,
#             "n_clusters": 6,
#             "corrupt_prob": 0.10,
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.5,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 1, "err_min": 15, "err_max": 20},  # type_2
#         },
#     },
# }


# save_folder = "../../data/GMM_Synth_two_cluster/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "dirty_classes_only",  # no_labels; joint_classes
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.10,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.10,
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.1,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 1, "err_min": 3, "err_max": 4},  # type_2
#         },
#     },
# }

# Synth Cluster Defs:
#     def __init__(
#         self,
#         n_samples=100,
#         n_clusters=4,
#         corrupt_prob=0.1,
#         n_features=2,
#         scale_density="same",
#         size_cluster="same",
#         std_scaler_cluster=1.0,
#         dist=0.25,
#         random_state=None,
#         noise_type=None,
#         noise_type_defs=None,
#     ):


# save_folder = "../../data/GMM_Synth_four_cluster/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes
#         "min_coverage": False,
#         "mc_mode": None,  # "stratisfied_v2"
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 4,
#             "corrupt_prob": 0.15,
#             "n_features": 2,
#             "std_scaler_cluster": 1.0,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "dist": 0.2,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 1, "err_min": 3, "err_max": 4},  # type_2
#         },
#     },
# }


# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth_two_cluster_betterlabels_type_2/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.20,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 0.8,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {
#                 "n_groups": 2,
#                 "err_min": 3.5,
#                 "err_max": 4.5,
#             },  # type_2
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth_two_cluster_betterlabels_type_1_v1/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.20,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.20,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_1",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 4, "err_max": 5},  # type_2
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)

# save_folder = "../../data/GMM_Synth_two_cluster_betterlabels_type_1/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.20,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 0.5,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_1",  # 'type_1'
#             "noise_type_defs": {
#                 "n_groups": 2,
#                 "scale_min_std": 1,
#                 "scale_max_std": 1.5,
#             },
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth_two_cluster_smalllabels_type_2/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.15,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3, "err_max": 4.5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


####


# save_folder = "../../data/GMM_Synth_one_cluster_3_dirty_smalllabels_type_2/"

# run_stats = {
#     "name": "simple_trial",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 1,
#             "corrupt_prob": 0.15,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 3, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


##### August 2020; 10th


## 1 Inlier Cluster; 3 Outlier Cluster; Type 2; Small Labels (less supervision)

# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "1clean_w_3dirty_type2_trust0.05_nolabels_corrup0.15",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 1,
#             "corrupt_prob": 0.15,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 3, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


## 1 Inlier Cluster; 3 Outlier Cluster; Type 2; Small Labels (less supervision), using stratisfied!


# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "1clean_w_3dirty_type2_trust0.05_stratisfied_corrup0.15",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 1,
#             "corrupt_prob": 0.15,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 3, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


## 2 inlier clusters; 2 outlier clusters, per clean cluster; Type 2; Small Labels

# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "1clean_w_3dirty_type2_trust0.05_stratisfied_corrup0.15",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 1,
#             "corrupt_prob": 0.15,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 3, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "2clean_w_2dirty_type2_trust0.05_nolabels_corrup0.20",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.20,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3, "err_max": 4.5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "2clean_w_2dirty_type2_trust0.05_stratisfied_corrup0.20",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.20,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3, "err_max": 4.5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "2clean_w_2dirty_type2_trust0.15_nolabels_corrup0.20",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.15,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 2,
#             "corrupt_prob": 0.20,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3, "err_max": 4.5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)

# Easy setting with lots of labels
# 2 inlier clusters; 2 outlier clusters, per clean cluster; Type 2; Large Labels / Stratisfied
save_folder = "../../data/GMM_Synth/"

run_stats = {
    "name": "2clean_w_2dirty_type2_trust0.45_stratisfied_corrup0.20",
    "train_size": 0.8,
    "valid_size": 0.1,
    "test_size": 0.1,
    "trusted_set": {
        "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
        "min_coverage": True,
        "mc_mode": "stratisfied_v2",
        "frac_trusted": 0.45,
        "y_class_on": True,
        "y_noise_lists_on": True,
    },
    "synth_data": {
        "type": "GaussianClusters",
        "defs": {
            "n_samples": 1000,
            "n_clusters": 2,
            "corrupt_prob": 0.20,  # 0.10
            "n_features": 2,
            "scale_density": "same",
            "size_cluster": "same",
            "std_scaler_cluster": 1.0,  # 1.0
            "dist": 0.25,  # 0.25
            "random_state": None,  # seed number
            "noise_type": "type_2",  # 'type_1'
            "noise_type_defs": {"n_groups": 2, "err_min": 3, "err_max": 4.5},
        },
    },
}

dirty_semi_supervised_synthethic(run_stats, save_folder)


# ## 4 inlier clusters; 2 outlier clusters, per clean cluster; Type 2; Small Labels

# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "4clean_w_2dirty_type2_trust0.05_nolabels_corrup0.30",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 4,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# ## 4 inlier clusters; 2 outlier clusters, per clean cluster; Type 2; Small Labels (stratisfied)

# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "4clean_w_2dirty_type2_trust0.05_stratisfied_corrup0.30",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "joint_classes",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": True,
#         "mc_mode": "stratisfied_v2",
#         "frac_trusted": 0.05,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 4,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)


# ## 4 inlier clusters; 2 outlier clusters, per clean cluster; Type 2; Medium Labels

# save_folder = "../../data/GMM_Synth/"

# run_stats = {
#     "name": "4clean_w_2dirty_type2_trust0.15_nolabels_corrup0.30",
#     "train_size": 0.8,
#     "valid_size": 0.1,
#     "test_size": 0.1,
#     "trusted_set": {
#         "use_labels": "no_labels",  # no_labels; joint_classes; dirty_classes_only
#         "min_coverage": False,
#         "mc_mode": None,
#         "frac_trusted": 0.15,
#         "y_class_on": True,
#         "y_noise_lists_on": True,
#     },
#     "synth_data": {
#         "type": "GaussianClusters",
#         "defs": {
#             "n_samples": 1000,
#             "n_clusters": 4,
#             "corrupt_prob": 0.30,  # 0.10
#             "n_features": 2,
#             "scale_density": "same",
#             "size_cluster": "same",
#             "std_scaler_cluster": 1.0,  # 1.0
#             "dist": 0.25,  # 0.25
#             "random_state": None,  # seed number
#             "noise_type": "type_2",  # 'type_1'
#             "noise_type_defs": {"n_groups": 2, "err_min": 3.5, "err_max": 5},
#         },
#     },
# }

# dirty_semi_supervised_synthethic(run_stats, save_folder)
