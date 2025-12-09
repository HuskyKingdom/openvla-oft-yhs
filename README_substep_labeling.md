# LIBERO Substep Labeling Tool

自动为LIBERO RLDS数据集的每个timestep标注动作类型（move/pick/place）。

## 功能特性

- ✅ 基于夹爪状态转换自动检测Pick和Place动作
- ✅ 基于末端执行器Z轴运动自适应扩展动作时间范围
- ✅ 支持多物体任务（多个pick-place循环）
- ✅ 处理所有4个LIBERO任务套件（spatial/object/goal/10）
- ✅ 生成详细的统计摘要

## 安装依赖

```bash
pip install tensorflow>=2.12.0 tensorflow-datasets>=4.9.0 numpy>=1.24.0 scipy>=1.10.0
```

## 使用方法

### 基本用法

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path substep_labels_output.json
```

### 处理特定套件

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path substep_labels_output.json \
    --suites libero_spatial_no_noops libero_object_no_noops
```

### 限制处理数量（用于快速测试）

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path substep_labels_output.json \
    --max_episodes 10 \
    --debug
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--apd_path` | APD_plans.json文件路径 | `APD_plans.json` |
| `--rlds_data_dir` | RLDS数据集根目录 | *必填* |
| `--output_path` | 输出JSON文件路径 | `substep_labels_output.json` |
| `--suites` | 要处理的套件列表 | 全部4个套件 |
| `--max_episodes` | 每个套件最大处理episode数 | 全部 |
| `--debug` | 启用调试日志 | False |

## 输出格式

输出JSON文件结构：

```json
{
  "libero_spatial": {
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": {
      "episode_0": {
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
        "total_timesteps": 150,
        "timestep_labels": [
          {"timestep": 0, "action": "move"},
          {"timestep": 1, "action": "move"},
          ...
          {"timestep": 45, "action": "pick"},
          ...
          {"timestep": 95, "action": "place"},
          ...
        ],
        "summary": {
          "pick_moments": [50],
          "place_moments": [100],
          "num_pick_place_cycles": 1,
          "pick_segments": [[45, 60]],
          "place_segments": [[95, 105]],
          "move_segments": [[0, 44], [61, 94], [106, 149]],
          "action_counts": {
            "move": 120,
            "pick": 15,
            "place": 15
          }
        }
      }
    }
  }
}
```

## 算法说明

### 动作检测策略

1. **Move/Reach**: 夹爪打开状态下的所有移动
2. **Pick**: 夹爪从打开到闭合的转换时刻及其扩展范围
   - 向前扩展：检测Z轴下降（接近物体）
   - 向后扩展：检测Z轴上升（提起物体）
3. **Place**: 夹爪从闭合到打开的转换时刻及其扩展范围
   - 向前扩展：检测移动到目标位置
   - 向后扩展：检测离开放置位置

### 核心参数

在脚本的`CONFIG`字典中可以调整以下参数：

```python
CONFIG = {
    "gripper_threshold": 0.04,           # 夹爪打开判定阈值
    "pick_expand_backward": 30,          # Pick向前扩展最大步数
    "pick_expand_forward": 20,           # Pick向后扩展最大步数
    "place_expand_backward": 20,         # Place向前扩展最大步数
    "place_expand_forward": 15,          # Place向后扩展最大步数
    "z_descent_threshold": -0.005,       # Z轴下降判定阈值
    "z_ascent_threshold": 0.01,          # Z轴上升判定阈值
    "movement_threshold": 0.05,          # 位移判定阈值
}
```

## 多物体任务支持

脚本自动支持多物体任务（如"put both moka pots on the stove"），会检测多次pick-place循环：

```json
{
  "summary": {
    "pick_moments": [50, 205],           // 两次pick
    "place_moments": [85, 245],          // 两次place
    "num_pick_place_cycles": 2,          // 2个完整循环
    "pick_segments": [[45, 60], [200, 215]],
    "place_segments": [[80, 90], [240, 250]]
  }
}
```

## 性能

- 单个episode处理时间：~0.1秒
- 完整4个suite（约400 episodes）：~1分钟

## 故障排除

### 问题：找不到RLDS数据集

```
Dataset not found: /path/to/libero_spatial_no_noops
```

**解决**：检查RLDS数据集路径是否正确，确保目录包含tfrecord文件。

### 问题：无法匹配instruction

```
No matching plan found for: 'some instruction'
```

**解决**：这是正常的警告，表示该instruction在APD_plans.json中没有对应的plan。脚本会继续处理并生成标注。

### 问题：内存不足

**解决**：使用`--max_episodes`参数限制每次处理的episode数量，分批处理。

## 开发者信息

- 基于RIPER-5协议开发
- 使用状态变化检测方法
- 核心信号：夹爪状态转换（最可靠的物理特征）

## 扩展

如需修改算法，主要修改以下函数：

- `expand_pick_range()`: 调整Pick动作范围扩展逻辑
- `expand_place_range()`: 调整Place动作范围扩展逻辑
- `detect_gripper_transitions()`: 调整夹爪转换检测逻辑

