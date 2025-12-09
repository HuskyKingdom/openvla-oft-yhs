# LIBERO Expert Episode Replay Tool

这个工具用于从LIBERO RLDS专家数据集中回放单个episode，并生成带有时间步、动作类型和APD计划步骤标注的视频。

## 功能特性

- 从RLDS数据集加载指定的专家演示episode
- 使用专家动作在LIBERO仿真环境中回放
- 在视频右上角实时显示：
  - 当前timestep编号
  - 当前动作类型（MOVE/PICK/PLACE）
  - 当前APD计划步骤描述
- 生成带标注的MP4视频

## 依赖项

确保已安装：
- LIBERO环境
- TensorFlow和TensorFlow Datasets
- OpenCV (cv2)
- imageio
- numpy

## 使用方法

### 基本用法

```bash
cd experiments/robot/libero

python run_expert_replay.py \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --substep_labels_path /path/to/substep_labels_output.json \
    --suite libero_spatial_no_noops \
    --episode_idx 0 \
    --output_dir ./expert_replay_videos
```

### 参数说明

#### 必需参数

- `--rlds_data_dir`: RLDS数据集的根目录路径
  - 示例: `/path/to/modified_libero_rlds`
  
- `--substep_labels_path`: substep labels JSON文件的路径
  - 这是由 `label_substeps.py` 生成的文件
  - 示例: `substep_labels_output.json`

- `--suite`: 任务套件名称
  - 可选值: `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`

- `--episode_idx`: 要回放的episode索引
  - 从0开始计数

#### 可选参数

- `--output_dir`: 输出视频的目录（默认: `./expert_replay_videos`）
- `--env_img_res`: 环境图像分辨率（默认: 256）
- `--num_steps_wait`: 开始时的等待步数（默认: 10）
- `--fps`: 视频帧率（默认: 30）

### 完整示例

```bash
# 回放 libero_spatial 套件的第0个episode
python run_expert_replay.py \
    --rlds_data_dir /data/modified_libero_rlds \
    --substep_labels_path ./substep_labels_output.json \
    --suite libero_spatial_no_noops \
    --episode_idx 0 \
    --output_dir ./videos \
    --env_img_res 256 \
    --num_steps_wait 10 \
    --fps 30

# 回放 libero_object 套件的第5个episode
python run_expert_replay.py \
    --rlds_data_dir /data/modified_libero_rlds \
    --substep_labels_path ./substep_labels_output.json \
    --suite libero_object_no_noops \
    --episode_idx 5 \
    --output_dir ./videos
```

## 输出

### 视频文件

视频文件将保存在指定的输出目录，文件名格式为：

```
{suite_name}__episode_{episode_idx}__{task_name}__success_{True/False}.mp4
```

示例:
```
libero_spatial_no_noops__episode_0__pick_up_the_black_bowl_on_the_plate__success_True.mp4
```

### 视频标注

每一帧的右上角会显示：

```
┌─────────────────────────────────┐
│ t=42                            │
│ PICK                            │
│ Pick up the object from the     │
│ table                           │
└─────────────────────────────────┘
```

- **第一行**: 当前timestep编号
- **第二行**: 动作类型（颜色编码）
  - MOVE: 青色
  - PICK: 绿色
  - PLACE: 橙色
- **后续行**: APD计划步骤描述（自动换行）

## 工作流程

1. **加载substep labels**: 从JSON文件加载预先生成的时间步标签
2. **加载RLDS episode**: 从RLDS数据集加载指定的expert episode
3. **提取数据**: 提取动作序列、任务指令和任务名称
4. **匹配标签**: 将RLDS数据与substep labels匹配
5. **初始化环境**: 创建LIBERO仿真环境并设置初始状态
6. **回放**: 使用专家动作在环境中执行，并记录每一帧
7. **标注**: 为每一帧添加timestep、action type和APD step信息
8. **保存**: 将标注后的帧保存为MP4视频

## 数据格式

### Substep Labels JSON 结构

```json
{
  "libero_spatial": {
    "task_name": {
      "episode_0": {
        "instruction": "pick up the black bowl",
        "total_timesteps": 150,
        "timestep_labels": [
          {
            "timestep": 0,
            "action": "move",
            "APD_step": "Move to the object"
          },
          {
            "timestep": 30,
            "action": "pick",
            "APD_step": "Pick up the object"
          },
          ...
        ],
        "summary": {...}
      }
    }
  }
}
```

## 注意事项

1. **Suite名称**: 确保suite名称包含 `_no_noops` 后缀（如果数据集使用该后缀）

2. **Episode索引**: Episode索引必须在数据集范围内，否则会报错

3. **专家动作格式**: 从RLDS直接读取的动作应该是环境可以直接使用的格式（gripper在 [-1, 1]范围）

4. **初始状态**: 脚本会自动从task suite获取对应episode的初始状态

5. **内存使用**: 所有帧都会在内存中保存直到写入视频，长episode可能需要较大内存

## 故障排除

### 问题: "Dataset not found"
- **解决**: 检查 `rlds_data_dir` 路径是否正确
- **验证**: 确认路径下存在 `{suite_name}/1.0.0/` 目录

### 问题: "No substep labels found"
- **解决**: 确保 substep_labels_output.json 包含对应suite、task和episode的标签
- **注意**: 如果没有标签，视频仍会生成，只是没有标注

### 问题: "Could not find matching task"
- **解决**: 检查RLDS episode中的instruction是否与LIBERO benchmark中的任务匹配
- **提示**: 查看日志中的task name和instruction

### 问题: Episode未成功完成
- **说明**: 专家数据可能包含失败的演示
- **文件名**: 视频文件名会标注 `success_False`

## 相关工具

- `label_substeps.py`: 生成substep labels的工具
- `run_libero_eval.py`: 使用模型进行LIBERO评估的脚本

## 许可证

请参考主项目的LICENSE文件。

