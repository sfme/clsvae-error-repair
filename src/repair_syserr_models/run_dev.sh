

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



##
#python -u main.py --save-on --output-folder ./testing_genlinearflow_y_semi_4_cluster --verbose-metrics-epoch --model-type semi_y_VAE_GenLinearFlow --number-epochs 400 --cuda-on --dataset-folder ../../data/GMM_Synth_four_cluster/simple_trial/ --semi-supervise

### run model CLSVAE for synthetic_shapes

# general options 
declare run_epochs=200
declare run_model_type="semi_y_CLSVAE"

declare run_save_folder="../../outputs/experiments_test/dummy_experiment/clsvae/" 
declare run_read_data_folder="../../data/examples_synthetic_shapes/corrupt_level_35_percent/run_1/" # 1 2 3
declare trust_set_name="10_samples_per_class" # 5; 10; 25, 50;

# train command
python -u main.py \
    --cuda-on \
    --save-on \
    --use-binary-img \
    --output-folder ${run_save_folder} \
    --verbose-metrics-epoch \
    --model-type ${run_model_type} \
    --number-epochs ${run_epochs} \
    --dataset-folder ${run_read_data_folder} \
    --semi-supervise \
    --use-sup-weights \
    --sup-loss-coeff 1000.0 \
    --kl-anneal \
    --kl-anneal-start 0.0 \
    --kl-anneal-stop 1.0 \
    --kl-anneal-ratio 0.5 \
    --kl-anneal-delay-epochs 10 \
    --sigma-eps-z-in 0.5 \
    --mean-eps-z-in 0.0 \
    --y-clean-prior 0.6 \
    --fixed-prior-z-clean 0.5 \
    --fixed-prior-z-dirty 5.0 \
    --dist-corr-reg \
    --reg-delay-n-epochs 10 \
    --reg-schedule-ratio 0.5 \
    --dist-corr-reg-coeff 100.0 \
    --trust-set-name ${trust_set_name}

