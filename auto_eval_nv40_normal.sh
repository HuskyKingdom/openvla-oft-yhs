#!/bin/bash
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
PRETRAINED_CHECKPOINT="ckpt/ckpoints/saga_h100_49"
TASK_LABEL_PREFIX="saga_rl_step49_h100_libero"
USE_EOS_DETECTION=False
EVAL_SCRIPT="experiments/robot/libero/run_libero_pro_eval_substep.py"
USE_PROPRIO=False
USE_L1_REGRESSION=False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
USE_SUBSTEP_DECOMPOSITION=False
USE_BDDL_LANGUAGE=True
AUTO_REGRESSION=False
NUM_IMAGES_IN_INPUT=1
SUBSTEP_COMPLETION_THRESHOLD=0.03



echo "Evaluating SUBSTEP ------------------------------"

# Reset all evaluation config flags to false before starting
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH


python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_object --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT


python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD\
  --task_suite_name libero_goal --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT



python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD \
  --task_suite_name libero_spatial --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT

python $EVAL_SCRIPT \
  --pretrained_checkpoint $PRETRAINED_CHECKPOINT --substep_completion_threshold $SUBSTEP_COMPLETION_THRESHOLD\
  --task_suite_name libero_10 --e_decoding False --save_video False --use_substep_decomposition $USE_SUBSTEP_DECOMPOSITION\
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_lan --use_eos_detection $USE_EOS_DETECTION --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT

