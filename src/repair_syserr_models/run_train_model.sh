#!/bin/bash

# choose setting /  model
declare model_to_run="clsvae_fashion_mnist" # clsvae_frey_faces ; clsvae_fashion_mnist ; clsvae_synthetic_shapes


if [ ${model_to_run} == "clsvae_synthetic_shapes" ]; then

    ### run model CLSVAE for synthetic_shapes

    # general options 
    declare run_epochs=200
    declare run_model_type="semi_y_CLSVAE"

    declare run_save_folder="../../outputs/experiments_test/dummy_experiment/clsvae/" 
    declare run_read_data_folder="../../data/examples_synthetic_shapes/corrupt_level_35_percent/run_1/" # 1 2 3
    declare trust_set_name="10_samples_per_class" # 5; 10; 25, 50;

    # train model command
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


elif [ ${model_to_run} == "clsvae_frey_faces" ]; then

    ### run model CLSVAE for frey_faces

    # general options
    declare run_epochs=300 
    declare run_model_type="semi_y_CLSVAE"

    declare run_save_folder="../../outputs/experiments_test/dummy_experiment/clsvae/" 
    declare run_read_data_folder="../../data/examples_frey_faces/corrupt_level_35_percent/run_1/" # 1 2 3
    declare trust_set_name="10_samples_per_class" # 5; 10; 25, 50;

    # train model command
    python -u main.py \
        --cuda-on \
        --save-on \
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
        --kl-anneal-ratio 0.2 \
        --kl-anneal-delay-epochs 10 \
        --sigma-eps-z-in 0.6 \
        --mean-eps-z-in 0.0 \
        --y-clean-prior 0.6 \
        --fixed-prior-z-clean 0.2 \
        --fixed-prior-z-dirty 5.0 \
        --dist-corr-reg \
        --reg-delay-n-epochs 10 \
        --reg-schedule-ratio 0.5 \
        --dist-corr-reg-coeff 1000.0 \
        --trust-set-name ${trust_set_name}

elif [ ${model_to_run} == "clsvae_fashion_mnist" ]; then

    ### run model CLSVAE for fashion_mnist

    # general options
    declare run_epochs=100
    declare run_model_type="semi_y_CLSVAE"

    declare run_save_folder="../../outputs/experiments_test/dummy_experiment/clsvae/" 
    declare run_read_data_folder="../../data/examples_fashion_mnist/corrupt_level_35_percent/run_1/" # 1 2 3
    declare trust_set_name="10_samples_per_class" # 5; 10; 25, 50;

    # train model command
    python -u main.py \
        --cuda-on \
        --save-on \
        --output-folder ${run_save_folder} \
        --verbose-metrics-epoch \
        --model-type ${run_model_type} \
        --number-epochs ${run_epochs} \
        --dataset-folder ${run_read_data_folder} \
        --semi-supervise \
        --use-sup-weights \
        --sup-loss-coeff 100.0 \
        --kl-anneal \
        --kl-anneal-start 0.0 \
        --kl-anneal-stop 1.0 \
        --kl-anneal-ratio 0.5 \
        --kl-anneal-delay-epochs 5 \
        --sigma-eps-z-in 0.1 \
        --mean-eps-z-in 0.0 \
        --y-clean-prior 0.6 \
        --fixed-prior-z-clean 0.2 \
        --fixed-prior-z-dirty 5.0 \
        --dist-corr-reg \
        --reg-delay-n-epochs 5 \
        --reg-schedule-ratio 0.5 \
        --dist-corr-reg-coeff 1000.0 \
        --trust-set-name ${trust_set_name}

fi








