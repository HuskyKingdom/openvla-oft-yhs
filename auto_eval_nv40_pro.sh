#!/bin/bash
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
PRETRAINED_CHECKPOINT="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--substep--substep_vla_scalling--150000_chkpt"
TASK_LABEL_PREFIX="apd_scalled"
USE_EOS_DETECTION=True
EVAL_SCRIPT="experiments/robot/libero/run_libero_pro_eval_substep.py"
USE_PROPRIO=True
USE_L1_REGRESSION=True
USE_SUBSTEP_DECOMPOSITION=True
USE_BDDL_LANGUAGE=True



echo "Evaluating SUBSTEP ------------------------------"


# swap
sed -i 's/use_environment: true/use_environment: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python $EVAL_SCRIPT  \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False  --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_swap --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_swap --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT  \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_spatial --e_decoding False --save_video False  --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_swap --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_swap --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE


# object
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_object --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_object --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_object --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_object --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE


# lan
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07\
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07\
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE


# task
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07\
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_task --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_task --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07\
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_task --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold 0.07 \
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_task --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE

sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
