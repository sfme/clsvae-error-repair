#!/bin/bash

# model_to_run="VAE" # "VAE"; "SEMI_Y_VAE"; "FULL_Y_CVAE"; "SEMI_Y_VAE_PART_SYS_OUT"

for model_to_run in "SEMI_Y_CCVAE" ; do # "VAE" "SEMI_Y_VAE" "FULL_Y_CVAE" "SEMI_Y_VAE_PART_SYS_OUT" "SEMI_Y_CCVAE"

    if [ ${model_to_run} == "VAE" ]; then

        ### VAE-L2

        # noise level
        for cur_noise_lvl in "15" "25" "35" "45" ; do

            # trustset level
            for cur_trustset_lvl in "5" "10" "25" "50" ; do

                # run number
                for cur_run in "1" "2" "3" "4" "5" ; do

                    # sh ./run_model.sh \
                    sbatch -J vae_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run} ./wrapper_cluster_run_model.sh \
                        --dataset "frey_faces" \
                        --model "VAE" \
                        --epochs "300" \
                        --run-number "${cur_run}" \
                        --noise-option "simple_systematic_shapes" \
                        --noise-level "${cur_noise_lvl}" \
                        --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
                        --kl-anneal-ratio "0.2" \
                        --kl-delay-epochs "10" \
                        --l2-reg-val "100.0"

                done
            done
        done


    elif [ ${model_to_run} == "SEMI_Y_VAE" ]; then

        ### SEMI_Y_VAE

        # noise level
        for cur_noise_lvl in "15" "25" "35" "45" ; do

            # trustset level
            for cur_trustset_lvl in "5" "10" "25" "50" ; do

                # run number
                for cur_run in "1" "2" "3" "4" "5" ; do

                    # sh ./run_model.sh \
                    sbatch -J semi_Y_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run} ./wrapper_cluster_run_model.sh \
                        --dataset "frey_faces" \
                        --model "semi_y_VAE" \
                        --epochs "300" \
                        --run-number "${cur_run}" \
                        --noise-option "simple_systematic_shapes" \
                        --noise-level "${cur_noise_lvl}" \
                        --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
                        --y-prior "0.6" \
                        --sup-loss-coeff "1000.0" \
                        --kl-anneal-ratio "0.2" \
                        --kl-delay-epochs "10" \
                        --z-y1-prior "0.6"

                done
            done
        done


    elif [ ${model_to_run} == "FULL_Y_CVAE" ]; then

        ### CVAE

        # noise level
        for cur_noise_lvl in "15" "25" "35" "45" ; do

            # trustset level
            for cur_trustset_lvl in "5" "10" "25" "50" ; do

                # run number
                for cur_run in "1" "2" "3" "4" "5" ; do

                    # sh ./run_model.sh \
                    sbatch -J cvae_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run} ./wrapper_cluster_run_model.sh \
                        --dataset "frey_faces" \
                        --model "full_y_CVAE" \
                        --epochs "300" \
                        --run-number "${cur_run}" \
                        --noise-option "simple_systematic_shapes" \
                        --noise-level "${cur_noise_lvl}" \
                        --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
                        --kl-anneal-ratio "0.2" \
                        --kl-delay-epochs "10" \
                        --z-y1-prior "0.2"

                done
            done
        done


    elif [ ${model_to_run} == "SEMI_Y_CCVAE" ]; then

        ### SEMI_Y_CCVAE

        # noise level
        for cur_noise_lvl in "15" "25" "35" "45" ; do

            # trustset level
            for cur_trustset_lvl in "5" "10" "25" "50" ; do

                # run number
                for cur_run in "1" "2" "3" "4" "5" ; do

                    # sh ./run_model.sh \
                    sbatch -J ccvae_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run} ./wrapper_cluster_run_model.sh \
                        --dataset "frey_faces" \
                        --model "semi_y_CCVAE" \
                        --epochs "300" \
                        --run-number "${cur_run}" \
                        --noise-option "simple_systematic_shapes" \
                        --noise-level "${cur_noise_lvl}" \
                        --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
                        --y-prior "0.6" \
                        --sup-loss-coeff "10000.0"

                done
            done
        done


    elif [ ${model_to_run} == "SEMI_Y_VAE_PART_SYS_OUT" ]; then

        ### SEMI_Y_VAE_PART_SYS_OUT (OUR MODEL)

        # noise level
        for cur_noise_lvl in "15" "25" "35" "45" ; do

            # trustset level
            for cur_trustset_lvl in "5" "10" "25" "50" ; do

                # run number
                for cur_run in "1" "2" "3" "4" "5" ; do

                    # type experiment: with / without dist_corr
                    for exper_type in "dist_corr_on" "dist_corr_off" ; do

                        if [[ ${exper_type} = "dist_corr_on" ]]; then
                            # sh ./run_model_bin.sh \
                            sbatch -J our_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run}_${exper_type} ./wrapper_cluster_run_model.sh \
                                --dataset "frey_faces" \
                                --model "semi_y_VAE_part_sys_out" \
                                --epochs "300" \
                                --run-number "${cur_run}" \
                                --noise-option "simple_systematic_shapes" \
                                --noise-level "${cur_noise_lvl}" \
                                --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
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

                        else
                            # sh ./run_model_bin.sh \
                            sbatch -J our_noise_${cur_noise_lvl}_trust_${cur_trustset_lvl}_run_${cur_run}_${exper_type} ./wrapper_cluster_run_model.sh \
                                --dataset "frey_faces" \
                                --model "semi_y_VAE_part_sys_out" \
                                --epochs "300" \
                                --run-number "${cur_run}" \
                                --noise-option "simple_systematic_shapes" \
                                --noise-level "${cur_noise_lvl}" \
                                --trust-set-name "${cur_trustset_lvl}_samples_per_class" \
                                --y-prior "0.6" \
                                --sup-loss-coeff "1000.0" \
                                --kl-anneal-ratio "0.2" \
                                --kl-delay-epochs "10" \
                                --sigma-noise-eps "0.6" \
                                --mean-noise-eps "0.0" \
                                --z-clean-prior "0.2" \
                                --z-dirty-prior "5.0" \
                                --type-experiment-name "dist_corr_off"

                        fi

                    done
                done
            done
        done

    fi

done
