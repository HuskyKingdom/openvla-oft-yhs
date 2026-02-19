## Research Guidence


我现在需要你帮我进行科研实验，包括想法开发，代码编写，实验验证，理论推理，最后形成 CVPR/ICML 级别的会议论文的完整方法。你需要首先仔细阅读我给你的研究方向和已有方法，基于方向和已有方法进行深度挖掘创新，完善一个更棒的方法。我会提供给你实验的远程服务器供你使用，方便你进行实验。

在你科研过程中，需要使用 task-status 工具向我报告进度。


### 研究方向和已有方法

参考本目录 `prev_research.md` 和 `substep_implementation_summary.md` 文件。

### 实验环境

你的所有代码都不允许在本地跑，本地仅进行代码编辑, `/home/yuhang/Desktop/openvla-oft-yhs` 本身就是一个git 仓库，你需要将本地的改动提交到 remote， 然后在远端服务器 pull 下来。


我提供两个远端实验环境：

1. 训练服务器 (amd cluster): 参考 tools 里面的 amd server. 使用时登陆进去后通过 `work` 指令进入工作目录，同时也会激活环境，通过 slurms 文件夹内的 `substep_plus.sh` 可以提交训练，你也可以生成新的 sh 文件提交想要的自定义任务。


2. 测试服务器 (NV 4090) (密码:asd69267822): 参考 tools 里面的 nv server. 仅用于测试, 使用时登陆进去后通过 `work` 指令进入工作目录， 然后使用 `conda activate vla_pro` 激活测试环境，并运行如下：

`export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO #libero pro`


常规测试脚本参考 `auto_eval_nv40_substep.sh`


### 要求

1. 你对主代码训练的改动应当在新的文件，我为你生成了 `/home/yuhang/Desktop/openvla-oft-yhs/experiments/robot/libero/run_libero_pro_eval_javas.py` 以及 `/home/yuhang/Desktop/openvla-oft-yhs/vla-scripts/finetune_substep_javas.py` 供你使用。

2. 训练模型完成后需要我来把 ckpt 转到 NV server 进行测试，这一步你让我来。

3. 尽量节省tokens。

4. 任务完成中需要撰写研究任务报告，包含尝试的方法，结果等。

