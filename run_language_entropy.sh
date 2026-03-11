#!/bin/bash
echo "Running Language Entropy Analysis (§4.3) ------------------------------"
# Path to your APD substep decomposition JSON
# e.g. generated during evaluation via SubstepManager, or exported from eval logs
APD_PLANS_PATH="${1:-./APD_plans.json}"
LLM_PATH="${2:-openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--substep--substep_vla--150000_chkpt/}"

mkdir -p ckpts analysis_outputs/language_entropy

echo "Using APD plans from: $APD_PLANS_PATH"
echo "Using tokenizer from: $LLM_PATH"

python experiments/analysis/language_entropy_analysis.py \
  --apd_plans_path "$APD_PLANS_PATH" \
  --tokenizer_name_or_path "$LLM_PATH" \
  --output_dir ./analysis_outputs/language_entropy \
  --top_k_tokens 50 \
  2>&1 | tee ckpts/language_entropy.txt

echo "Language Entropy analysis done. Results in ckpts/language_entropy.txt and analysis_outputs/language_entropy/"

