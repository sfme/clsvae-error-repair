#!/bin/bash

sbatch -J our_noise_45_trust_25_run_2_on ./wrapper_cluster_run_model.sh \
    --dataset "frey_faces" \
    --model "semi_y_VAE_part_sys_out" \
    --epochs "300" \
    --run-number "2" \
    --noise-option "simple_systematic_shapes" \
    --noise-level "45" \
    --trust-set-name "25_samples_per_class" \
    --y-prior "0.6" \
    --sup-loss-coeff "1000.0" \
    --kl-anneal-ratio "0.2" \
    --kl-delay-epochs "10" \
    --sigma-noise-eps "0.6" \
    --mean-noise-eps "0.0" \
    --z-clean-prior "0.2" \
    --z-dirty-prior "5.0" \
    --dist-corr-reg true \
    --reg-delay-epochs "10" \
    --reg-schedule-ratio "0.5" \
    --dist-corr-reg-coeff "1000.0" \
    --type-experiment-name "dist_corr_on"

sbatch -J our_noise_45_trust_25_run_1_off ./wrapper_cluster_run_model.sh \
    --dataset "frey_faces" \
    --model "semi_y_VAE_part_sys_out" \
    --epochs "300" \
    --run-number "1" \
    --noise-option "simple_systematic_shapes" \
    --noise-level "45" \
    --trust-set-name "25_samples_per_class" \
    --y-prior "0.6" \
    --sup-loss-coeff "1000.0" \
    --kl-anneal-ratio "0.2" \
    --kl-delay-epochs "10" \
    --sigma-noise-eps "0.6" \
    --mean-noise-eps "0.0" \
    --z-clean-prior "0.2" \
    --z-dirty-prior "5.0"