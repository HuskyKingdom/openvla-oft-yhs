# 快速开始指南

## 1. 准备工作

确保你有以下文件和数据：

- ✅ `APD_plans.json` - 已在项目根目录
- ✅ `label_substeps.py` - 主脚本
- ⚠️ RLDS数据集 - 需要设置路径

## 2. 设置RLDS数据集路径

编辑你的命令或脚本，设置正确的数据集路径：

```bash
RLDS_DATA_DIR="/path/to/your/modified_libero_rlds"
```

数据集目录应包含以下子目录：
```
modified_libero_rlds/
├── libero_spatial_no_noops/
├── libero_object_no_noops/
├── libero_goal_no_noops/
└── libero_10_no_noops/
```

## 3. 快速测试

先用少量数据测试：

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path test_output.json \
    --max_episodes 5 \
    --debug
```

预期输出：
```
============================================================
LIBERO Substep Labeling Tool
============================================================
Loading APD plans from APD_plans.json
  Suite 'spatial': 10 instructions
  Suite 'object': 10 instructions
  Suite 'goal': 10 instructions
  Suite 'long': 10 instructions

Processing 4 suites:
  - libero_spatial_no_noops
  - libero_object_no_noops
  - libero_goal_no_noops
  - libero_10_no_noops

============================================================
Processing suite: libero_spatial_no_noops
============================================================
...
```

## 4. 检查输出

查看生成的 `test_output.json` 文件：

```bash
# Linux/Mac
head -n 50 test_output.json

# Windows PowerShell
Get-Content test_output.json -Head 50
```

或使用Python查看：

```python
import json
with open('test_output.json', 'r') as f:
    data = json.load(f)
    
# 查看结构
for suite, tasks in data.items():
    print(f"Suite: {suite}")
    for task, episodes in tasks.items():
        print(f"  Task: {task} ({len(episodes)} episodes)")
```

## 5. 完整运行

如果测试成功，运行完整处理：

```bash
python label_substeps.py \
    --apd_path APD_plans.json \
    --rlds_data_dir /path/to/modified_libero_rlds \
    --output_path substep_labels_output.json
```

## 6. 验证结果

使用以下Python脚本快速验证结果：

```python
import json

with open('substep_labels_output.json', 'r') as f:
    data = json.load(f)

# 统计
total_suites = len(data)
total_tasks = sum(len(tasks) for tasks in data.values())
total_episodes = sum(
    len(episodes) 
    for tasks in data.values() 
    for episodes in tasks.values()
)

print(f"Total suites: {total_suites}")
print(f"Total tasks: {total_tasks}")
print(f"Total episodes: {total_episodes}")

# 检查动作分布
action_totals = {"move": 0, "pick": 0, "place": 0}
for suite_data in data.values():
    for task_data in suite_data.values():
        for episode_data in task_data.values():
            counts = episode_data['summary']['action_counts']
            for action, count in counts.items():
                action_totals[action] += count

print(f"\nAction distribution:")
for action, count in action_totals.items():
    print(f"  {action}: {count}")
```

## 7. 常见问题

### Q: 提示找不到数据集
```
Dataset not found: /path/to/libero_spatial_no_noops
```

**A**: 检查路径是否正确，确保目录存在且包含tfrecord文件。

### Q: 内存不足

**A**: 使用 `--max_episodes` 参数限制处理数量：
```bash
python label_substeps.py ... --max_episodes 50
```

### Q: 想只处理特定suite

**A**: 使用 `--suites` 参数：
```bash
python label_substeps.py ... --suites libero_spatial_no_noops
```

### Q: 需要调试信息

**A**: 添加 `--debug` 参数：
```bash
python label_substeps.py ... --debug
```

## 8. 下一步

- 查看 `README_substep_labeling.md` 了解详细文档
- 运行 `python test_labeling.py` 执行单元测试
- 查看 `example_usage.sh` 了解更多使用示例

## 需要帮助？

检查以下文件：
- `README_substep_labeling.md` - 完整文档
- `label_substeps.py` - 源代码（有详细注释）
- `test_labeling.py` - 测试用例和示例

