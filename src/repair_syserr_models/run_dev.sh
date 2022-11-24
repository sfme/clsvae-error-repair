

#python -u main.py --save-on --output-folder ./dummy_gmm_v2 --verbose-metrics-epoch --model-type VAE_GMM_v2 --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth/simple_trial/ --semi-supervise


# --output-folder ../output_testing/

#python -u main.py --verbose-metrics-epoch --model-type modVAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth/simple_trial/ --semi-supervise


#python -u main.py --save-on --output-folder ./dummy_basic_y_semi_static --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 1000 --cuda-on --dataset-folder ../../data/GMM_Synth/simple_trial/ --semi-supervise

# python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v2_static_prior --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_type1/simple_trial/ --semi-supervise

# python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v3_static_prior --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_2/simple_trial/ --semi-supervise

# python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v4_static --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_3/simple_trial/ --semi-supervise

#python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v5 --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_4/simple_trial/ --semi-supervise

# python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v6 --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_5_high_labels/simple_trial/ --semi-supervise

# python -u main.py --save-on --output-folder ./dummy_basic_y_semi_v6_static --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 500 --cuda-on --dataset-folder ../../data/GMM_Synth_5_high_labels/simple_trial/ --semi-supervise

#python -u main.py --save-on --output-folder ./dummy_basic_y_semi_4_cluster --verbose-metrics-epoch --model-type semi_y_VAE --number-epochs 400 --cuda-on --dataset-folder ../../data/GMM_Synth_four_cluster/simple_trial/ --semi-supervise

python -u main.py --save-on --output-folder ./testing_genlinearflow_y_semi_4_cluster --verbose-metrics-epoch --model-type semi_y_VAE_GenLinearFlow --number-epochs 400 --cuda-on --dataset-folder ../../data/GMM_Synth_four_cluster/simple_trial/ --semi-supervise
