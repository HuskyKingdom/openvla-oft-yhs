#!/bin/bash

# Example usage script for LIBERO substep labeling tool
# 示例：如何使用LIBERO substep标注工具

echo "========================================"
echo "LIBERO Substep Labeling - Example Usage"
echo "========================================"

# ============================================
# 配置参数 - 请根据实际情况修改
# ============================================

# APD plans文件路径
APD_PATH="APD_plans.json"

# RLDS数据集根目录（需要修改为实际路径）
RLDS_DATA_DIR="/path/to/modified_libero_rlds"  # TODO: 修改此路径

# 输出文件路径
OUTPUT_PATH="substep_labels_output.json"

# ============================================
# 示例1: 快速测试（处理少量episodes）
# ============================================
echo ""
echo "Example 1: Quick test with 5 episodes per suite"
echo "------------------------------------------------"

python label_substeps.py \
    --apd_path "${APD_PATH}" \
    --rlds_data_dir "${RLDS_DATA_DIR}" \
    --output_path "test_output.json" \
    --max_episodes 5 \
    --debug

# ============================================
# 示例2: 处理单个suite
# ============================================
echo ""
echo "Example 2: Process only spatial suite"
echo "--------------------------------------"

python label_substeps.py \
    --apd_path "${APD_PATH}" \
    --rlds_data_dir "${RLDS_DATA_DIR}" \
    --output_path "spatial_labels.json" \
    --suites libero_spatial_no_noops

# ============================================
# 示例3: 处理多个特定suites
# ============================================
echo ""
echo "Example 3: Process spatial and object suites"
echo "---------------------------------------------"

python label_substeps.py \
    --apd_path "${APD_PATH}" \
    --rlds_data_dir "${RLDS_DATA_DIR}" \
    --output_path "spatial_object_labels.json" \
    --suites libero_spatial_no_noops libero_object_no_noops

# ============================================
# 示例4: 完整处理所有suites
# ============================================
echo ""
echo "Example 4: Process all suites (full dataset)"
echo "---------------------------------------------"

python label_substeps.py \
    --apd_path "${APD_PATH}" \
    --rlds_data_dir "${RLDS_DATA_DIR}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "========================================"
echo "Examples completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. 修改 RLDS_DATA_DIR 为您的实际数据集路径"
echo "2. 选择一个示例命令运行"
echo "3. 检查输出的JSON文件"

