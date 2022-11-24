#!/bin/bash

PARSED_ARGUMENTS=$(getopt -a -n run_model -o '' --long dataset:,model:,savepath:,readpath:,y-prior:,epochs:,run-number:,noise-option:,noise-level:,trust-set-name:,sup-loss-coeff:,kl-anneal-ratio:,kl-delay-epochs:,sigma-noise-eps:,mean-noise-eps:,mmd-reg-coeff:,mmd-reg-cap:,l2-reg-val:,z-y1-prior:,z-y0-prior:,z-clean-prior:,z-dirty-prior:,dist-corr-reg:,reg-delay-epochs:,reg-schedule-ratio:,dist-corr-reg-coeff:,type-experiment-name: -- "$@")

VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
    usage
fi

echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
eval set -- "$PARSED_ARGUMENTS"

SAVEPATH=""
READPATH=""


while :
do
    case "$1" in
        --dataset)              DATASET="$2"                ; shift 2 ;;
        --model)                MODEL="$2"                  ; shift 2 ;;
        --savepath)             SAVEPATH="$2"               ; shift 2 ;;
        --readpath)             READPATH="$2"               ; shift 2 ;;
        --y-prior)              Y_PRIOR_VAL="$2"            ; shift 2 ;;
        --epochs)               EPOCHS="$2"                 ; shift 2 ;;
        --run-number)           RUN_NUMBER="$2"             ; shift 2 ;;
        --noise-option)         NOISE_OPTION="$2"           ; shift 2 ;;
        --noise-level)          NOISE_LEVEL="$2"            ; shift 2 ;;
        --trust-set-name)       TRUST_SET_NAME="$2"         ; shift 2 ;;
        --sup-loss-coeff)       SUP_LOSS_COEFF="$2"         ; shift 2 ;;
        --kl-anneal-ratio)      KL_ANNEAL_RATIO="$2"        ; shift 2 ;;
        --kl-delay-epochs)      KL_DELAY_EPOCHS="$2"        ; shift 2 ;;
        --sigma-noise-eps)      SIGMA_NOISE_EPS="$2"        ; shift 2 ;;
        --mean-noise-eps)       MEAN_NOISE_EPS="$2"         ; shift 2 ;;
        --mmd-reg-coeff)        MMD_REG_COEFF="$2"          ; shift 2 ;;
        --mmd-reg-cap)          MMD_REG_CAP="$2"            ; shift 2 ;;
        --l2-reg-val)           L2_REG_VAL="$2"             ; shift 2 ;;
        --z-y1-prior)           Z_Y1_PRIOR="$2"             ; shift 2 ;;
        --z-y0-prior)           Z_Y0_PRIOR="$2"             ; shift 2 ;;
        --z-clean-prior)        Z_CLEAN_PRIOR="$2"          ; shift 2 ;;
        --z-dirty-prior)        Z_DIRTY_PRIOR="$2"          ; shift 2 ;;
        --dist-corr-reg)        DIST_CORR_REG="$2"          ; shift 2 ;;
        --reg-delay-epochs)     REG_DELAY_EPOCHS="$2"       ; shift 2 ;;
        --reg-schedule-ratio)   REG_SCHEDULE_RATIO="$2"     ; shift 2 ;;
        --dist-corr-reg-coeff)  DIST_CORR_REG_COEFF="$2"    ; shift 2 ;;
        --type-experiment-name) TYPE_EXPERIMENT_NAME="$2"   ; shift 2 ;;

        --) shift; break ;;

        *) echo "Unexpected option: $1 - this should not happen."
            usage ;;
    esac
done


if [ -z "${SAVEPATH}" ]; then

    if [ -z "${TYPE_EXPERIMENT_NAME}" ]; then

        declare save_folder_exp="${DATASET}/${NOISE_OPTION}/corrupt_level_${NOISE_LEVEL}_percent/run_${RUN_NUMBER}/trustset_level_${TRUST_SET_NAME}/default/"
        declare save_folder_exp="../../../outputs/base_experiments_v2/balanced/${save_folder_exp}"

    else
        declare save_folder_exp="${DATASET}/${NOISE_OPTION}/corrupt_level_${NOISE_LEVEL}_percent/run_${RUN_NUMBER}/trustset_level_${TRUST_SET_NAME}/${MODEL}_${TYPE_EXPERIMENT_NAME}/"
        declare save_folder_exp="../../../outputs/base_experiments_v2/balanced/${save_folder_exp}"

    fi

else
    declare save_folder_exp="${SAVEPATH}"

fi

if [ -z "${READPATH}" ]; then
    declare read_folder_exp="${DATASET}/${NOISE_OPTION}/corrupt_level_${NOISE_LEVEL}_percent/run_${RUN_NUMBER}/"
    declare read_folder_exp="../../../data/base_experiments_v2/balanced/${read_folder_exp}"

else
    declare read_folder_exp="${READPATH}"

fi



if [ -z "${Z_Y1_PRIOR}" ]; then
    Z_Y1_PRIOR="1.0"
fi

if [ -z "${Z_Y0_PRIOR}" ]; then
    Z_Y0_PRIOR="5.0"
fi



if [ -z "${MEAN_NOISE_EPS}" ]; then
    MEAN_NOISE_EPS="0.0"
fi

if [ -z "${SIGMA_NOISE_EPS}" ]; then
    SIGMA_NOISE_EPS="0.5"
fi

if [ -z "${Z_CLEAN_PRIOR}" ]; then
    Z_CLEAN_PRIOR="0.5"
fi

if [ -z "${Z_DIRTY_PRIOR}" ]; then
    Z_DIRTY_PRIOR="5.0"
fi

if [ -z "${DIST_CORR_REG}" ]; then
    dist_corr_reg=false
else
    dist_corr_reg=${DIST_CORR_REG}
fi


echo -e "\n\n"
echo -e "${read_folder_exp} \n"
echo -e "${save_folder_exp} \n\n\n"

#### RUN MODEL ####

if [ ${MODEL} == "VAE" ]; then

    python ../../smm_models/main.py \
        --cuda-on \
        --save-on \
        --output-folder ${save_folder_exp} \
        --verbose-metrics-epoch \
        --model-type ${MODEL} \
        --number-epochs ${EPOCHS} \
        --dataset-folder ${read_folder_exp} \
        --kl-anneal \
        --kl-anneal-start 0.0 \
        --kl-anneal-stop 1.0 \
        --kl-anneal-ratio ${KL_ANNEAL_RATIO} \
        --kl-anneal-delay-epochs ${KL_DELAY_EPOCHS} \
        --l2-reg ${L2_REG_VAL} \
        --trust-set-name ${TRUST_SET_NAME}

elif [ ${MODEL} == "semi_y_VAE" ]; then

    python ../../smm_models/main.py \
        --cuda-on \
        --save-on \
        --output-folder ${save_folder_exp} \
        --verbose-metrics-epoch \
        --model-type ${MODEL} \
        --number-epochs ${EPOCHS} \
        --dataset-folder ${read_folder_exp} \
        --semi-supervise \
        --sup-loss-coeff ${SUP_LOSS_COEFF} \
        --kl-anneal \
        --kl-anneal-start 0.0 \
        --kl-anneal-stop 1.0 \
        --kl-anneal-ratio ${KL_ANNEAL_RATIO} \
        --kl-anneal-delay-epochs ${KL_DELAY_EPOCHS} \
        --fixed-prior-zy1-sigma ${Z_Y1_PRIOR} \
        --y-clean-prior ${Y_PRIOR_VAL} \
        --use-sup-weights \
        --trust-set-name ${TRUST_SET_NAME}

elif [ ${MODEL} == "semi_y_VAE_part_sys_out" ]; then

    if [ "${dist_corr_reg}" = "true" ]; then
        python -u ../../smm_models/main.py \
            --cuda-on \
            --save-on \
            --output-folder ${save_folder_exp} \
            --verbose-metrics-epoch \
            --model-type ${MODEL} \
            --number-epochs ${EPOCHS} \
            --dataset-folder ${read_folder_exp} \
            --semi-supervise \
            --sup-loss-coeff ${SUP_LOSS_COEFF} \
            --kl-anneal \
            --kl-anneal-start 0.0 \
            --kl-anneal-stop 1.0 \
            --kl-anneal-ratio ${KL_ANNEAL_RATIO} \
            --kl-anneal-delay-epochs ${KL_DELAY_EPOCHS} \
            --sigma-eps-z-in ${SIGMA_NOISE_EPS} \
            --mean-eps-z-in ${MEAN_NOISE_EPS} \
            --y-clean-prior ${Y_PRIOR_VAL} \
            --use-sup-weights \
            --fixed-prior-z-clean ${Z_CLEAN_PRIOR} \
            --fixed-prior-z-dirty ${Z_DIRTY_PRIOR} \
            --dist-corr-reg \
            --reg-delay-n-epochs ${REG_DELAY_EPOCHS} \
            --reg-schedule-ratio ${REG_SCHEDULE_RATIO} \
            --dist-corr-reg-coeff ${DIST_CORR_REG_COEFF} \
            --trust-set-name ${TRUST_SET_NAME}

    else
        python -u ../../smm_models/main.py \
            --cuda-on \
            --save-on \
            --output-folder ${save_folder_exp} \
            --verbose-metrics-epoch \
            --model-type ${MODEL} \
            --number-epochs ${EPOCHS} \
            --dataset-folder ${read_folder_exp} \
            --semi-supervise \
            --sup-loss-coeff ${SUP_LOSS_COEFF} \
            --kl-anneal \
            --kl-anneal-start 0.0 \
            --kl-anneal-stop 1.0 \
            --kl-anneal-ratio ${KL_ANNEAL_RATIO} \
            --kl-anneal-delay-epochs ${KL_DELAY_EPOCHS} \
            --sigma-eps-z-in ${SIGMA_NOISE_EPS} \
            --mean-eps-z-in ${MEAN_NOISE_EPS} \
            --y-clean-prior ${Y_PRIOR_VAL} \
            --use-sup-weights \
            --fixed-prior-z-clean ${Z_CLEAN_PRIOR} \
            --fixed-prior-z-dirty ${Z_DIRTY_PRIOR} \
            --trust-set-name ${TRUST_SET_NAME}

    fi


elif [ ${MODEL} == "semi_y_CCVAE" ]; then

    python -u ../../smm_models/main.py \
        --cuda-on \
        --save-on \
        --output-folder ${save_folder_exp} \
        --verbose-metrics-epoch \
        --model-type ${MODEL} \
        --number-epochs ${EPOCHS} \
        --dataset-folder ${read_folder_exp} \
        --semi-supervise \
        --y-clean-prior ${Y_PRIOR_VAL} \
        --lr "1e-3" \
        --q-y-x-coeff ${SUP_LOSS_COEFF} \
        --trust-set-name ${TRUST_SET_NAME}


elif [ ${MODEL} == "full_y_CVAE" ]; then

    python -u ../../smm_models/main.py \
        --cuda-on \
        --save-on \
        --output-folder ${save_folder_exp} \
        --verbose-metrics-epoch \
        --model-type ${MODEL} \
        --number-epochs ${EPOCHS} \
        --dataset-folder ${read_folder_exp} \
        --kl-anneal \
        --kl-anneal-start 0.0 \
        --kl-anneal-stop 1.0 \
        --kl-anneal-ratio ${KL_ANNEAL_RATIO} \
        --kl-anneal-delay-epochs ${KL_DELAY_EPOCHS} \
        --fixed-prior-zy1-sigma ${Z_Y1_PRIOR} \
        --use-q-z-y \
        --trust-set-name ${TRUST_SET_NAME}

fi
