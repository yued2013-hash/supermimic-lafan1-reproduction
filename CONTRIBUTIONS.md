# 实现来源与实验贡献

本文区分上游算法实现、独立编写的实验工具及实际完成的实验。不将方法复现表述为原创算法。

## 上游继承

固定实现：[`Shimingyi/allmimic@cb334ff`](https://github.com/Shimingyi/allmimic/tree/cb334ff674b2f57248f33f8c21b7fb526fbdf8cc)。

| 上游文件 | 使用的功能 |
| :-- | :-- |
| `allmimic/train.py` | replay采样、世界模型拟合、想象轨迹/BPTT、critic、优化器、checkpoint与训练循环 |
| `allmimic/networks.py` | 世界模型、策略、critic及状态转移运算 |
| `allmimic/sim_env.py`、`task.py` | MuJoCo-Warp物理环境、动作映射、观测、奖励和终止 |
| `allmimic/motion_lib.py` | 动作库表示与索引 |
| `scripts/prepare_motions.py` | LAFAN1动作格式转换与重采样 |
| `tools/eval_allmimic.py`、`allmimic/evaluate.py` | 独立FP32评估与完整动作执行协议 |

已发布seed0实验没有修改上述训练算法。`recommended` preset的残差动作、关节权重、优化稳定性设置等均来自上游，不能计为本项目提出的改进。

## 独立编写并公开的文件

| 文件 | 功能与边界 |
| :-- | :-- |
| `scripts/train.sh` | 固定上游commit、拒绝复用输出路径、提供实际实验启动参数；调用上游训练入口 |
| `scripts/evaluate.sh` | 固定commit、检查动作库与输出路径、调用独立FP32评估 |
| `scripts/download_lafan1.py` | 按固定数据revision下载40份CSV并逐文件验证SHA256 |
| `scripts/verify_results.py` | 无GPU核验完整步数、动作唯一性、回报统计、有限日志、训练计数与可选checkpoint哈希 |
| `scripts/plot_learning_curves.py` | 从原始记录重绘评估回报、失败数和世界模型误差 |
| `scripts/prepare_horizon_source.py` | 按固定源码SHA创建独立副本，仅统一策略loss除数，记录改动前后哈希 |
| `scripts/prepare_wm_diagnostic.py` | 固定抽样与来源校验，转换MuJoCo局部根角速度，构建共同诊断窗口 |
| `scripts/wm_diagnostic_runtime.py`、`evaluate_wm_diagnostic.py` | 调用上游世界模型/运动学/误差函数，提供CPU/GPU eager加载及统一开放环误差评估；不重新实现训练框架 |
| `scripts/check_loss_scaling.py` | 真实冻结世界模型上的测试目标autograd、裁剪和方向有限差分检查；非实际actor-critic/GPU编译验证 |
| `scripts/run_horizon.py` | 默认只打印配置；显式执行时记录运行条件与总耗时，隔离输出并限制单组运行时长 |
| `tests/test_ablation_preparation.py` | 固定除数补丁、重复/错误源拒绝、诊断坐标及误用训练数据类型检查 |
| `tests/test_evidence.py` | 验证真实记录通过，并拒绝重复动作、截短序列、错误统计和非有限回报 |
| `docs/index.html` | 视频、曲线、逐动作指标的静态展示与筛选 |
| `docs/experiment.md`、`docs/reproduce.md` | 实测条件、指标定义、操作步骤和验证边界 |

## 实际完成的实验工程

- 检查依赖与GPU环境，保留独立Python环境及实际版本记录。
- 固定数据版本并校验40条动作；原始数据共441120帧（50Hz处理后）。
- 运行随机初始化的联合训练，保存最终及7个中间checkpoint，加载最终策略进行独立评估。
- 核对40条长序列均执行完整步数，区分训练内bf16和独立fp32评估。
- 导出40条完整50Hz物理轨迹及视频，保留pre-reset终态，进行本地备份与哈希校验；公开仓库提供代表视频与全部逐项指标。
- 记录成功与未完成检查：专门compiled-BPTT一致性测试超时仍作为缺口保留。

## 实验决策与尚未完成的部分

初轮采用RTX4090、seed0、4096环境、recommended preset和2小时累计训练预算；运行参数可由 [`checkpoint_metadata.json`](results/lafan1_seed0/checkpoint_metadata.json)核对。

H=16/32训练对照尚未运行；固定除数的独立本地副本与真实checkpoint的CPU诊断已完成，见 [`docs/cpu_preparation.md`](docs/cpu_preparation.md)。额外seed1/2尚未训练。代码准备、诊断测量和新训练性能结论分别标注。

没有实现或验证未见动作泛化、跨seed稳定性、实机部署或相对其他算法的性能提升。训练源码、原始数据、机器人资产与权重的公开再分发仍受各自条款约束。
