#!/bin/bash
echo "Running KL Divergence Analysis (§4.2) ------------------------------"
CKPT_APD="openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--substep--substep_vla--150000_chkpt/"
CKPT_OFT="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"

mkdir -p ckpts

# APD model — libero_object
echo "KL Analysis: APD on libero_object ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_APD \
  --task_suite_name libero_object \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label APD \
  --output_dir ./analysis_outputs/kl_divergence/apd_object \
  2>&1 | tee ckpts/kl_apd_object.txt

# APD model — libero_goal
echo "KL Analysis: APD on libero_goal ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_APD \
  --task_suite_name libero_goal \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label APD \
  --output_dir ./analysis_outputs/kl_divergence/apd_goal \
  2>&1 | tee ckpts/kl_apd_goal.txt

# OFT baseline — libero_object
echo "KL Analysis: OFT on libero_object ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_object \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label OFT \
  --output_dir ./analysis_outputs/kl_divergence/oft_object \
  2>&1 | tee ckpts/kl_oft_object.txt

# OFT baseline — libero_goal
echo "KL Analysis: OFT on libero_goal ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label OFT \
  --output_dir ./analysis_outputs/kl_divergence/oft_goal \
  2>&1 | tee ckpts/kl_oft_goal.txt

# APD vs OFT comparison — libero_object
echo "KL Analysis: APD vs OFT comparison on libero_object ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_APD \
  --compare_checkpoint $CKPT_OFT \
  --task_suite_name libero_object \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label APD_vs_OFT \
  --output_dir ./analysis_outputs/kl_divergence/comparison_object \
  2>&1 | tee ckpts/kl_comparison_object.txt

# APD vs OFT comparison — libero_goal
echo "KL Analysis: APD vs OFT comparison on libero_goal ------------------------------"
python experiments/analysis/kl_divergence_analysis.py \
  --pretrained_checkpoint $CKPT_APD \
  --compare_checkpoint $CKPT_OFT \
  --task_suite_name libero_goal \
  --num_samples_per_task 200 \
  --use_l1_regression True \
  --use_proprio True \
  --num_images_in_input 2 \
  --checkpoint_label APD_vs_OFT \
  --output_dir ./analysis_outputs/kl_divergence/comparison_goal \
  2>&1 | tee ckpts/kl_comparison_goal.txt

echo "KL Divergence analysis done. Results in ckpts/kl_*.txt and analysis_outputs/kl_divergence/"

