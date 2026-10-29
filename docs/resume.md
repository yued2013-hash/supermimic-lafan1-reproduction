# 具身智能实习简历项目描述

## 推荐版本：放入正式简历

**基于可微世界模型的人形机器人动作模仿｜SuperMimic复现**　2026.10  
技术栈：Python、PyTorch、MuJoCo Warp、mjlab、世界模型、BPTT、强化学习

- **方法复现：** 基于第三方实现跑通世界模型、策略与critic的联合训练，梳理8步自回归动力学拟合、32步想象轨迹BPTT及TD(λ)价值学习流程，区分策略残差动作与物理绝对setpoint输入。
- **实验结果：** 在单卡RTX4090、4096并行环境下从随机初始化训练，约2小时累计训练计时完成1.26亿真实仿真环境步；单seed独立FP32评估覆盖LAFAN1全部40条训练库长动作，取得0失败、0.9271平均归一化回报。
- **实验工程：** 完成40条动作的数据校验与50Hz处理、checkpoint保存/加载、逐动作完整评估及物理轨迹/视频导出，建立包含固定代码commit、数据revision、环境版本和SHA256的可复验记录。

项目：https://github.com/yued2013-hash/supermimic-lafan1-reproduction  
展示：https://yued2013-hash.github.io/supermimic-lafan1-reproduction/

## 精简版本：简历空间较少时

> 基于第三方实现复现SuperMimic的可微世界模型与BPTT策略训练，在单卡RTX4090上以约2小时累计训练计时完成LAFAN1 40条训练库长动作跟踪，单seed独立FP32评估0失败、平均归一化回报0.9271；完成数据、checkpoint、逐动作指标与完整物理回放的可复验归档。

## English

**Humanoid Motion Imitation with Differentiable World Models — SuperMimic Reproduction**

- Reproduced a third-party training pipeline combining 8-step autoregressive dynamics learning, 32-step imagined-rollout BPTT policy optimization, and TD-lambda critic fitting in PyTorch and MuJoCo Warp.
- Trained from random initialization on a single RTX 4090 with 4,096 parallel environments, collecting 125.9M simulator transitions in approximately two accumulated training hours. Independent FP32 evaluation completed all 40 in-library LAFAN1 sequences with zero failures and a mean normalized return of 0.9271 (single seed).
- Built reproducibility records covering data checksums, pinned code/data revisions, environment versions, checkpoint loading, per-sequence metrics, and complete physical rollouts and videos.

## 30秒面试介绍

我复现的是一个基于学习动力学的动作模仿流程。物理仿真先收集状态转移，世界模型学习预测动作带来的状态变化，策略再在模型中想象32步，通过BPTT优化回报。我用4090完成了40条LAFAN1长动作的从零训练与完整评估，独立FP32平均回报0.9271且没有失败动作。我的工作重点是把环境、数据、训练、评估和记录串成可核验闭环；当前正在探索Dex5迁移，先处理接触模型和稳定持笔，尚未宣称完成方法改进。

## 书写边界

可以写：第三方实现复现、随机初始化训练、单seed/训练库评估、实际测得数字、独立核验与归档。

不要写：原创提出SuperMimic、独立从零实现整个训练框架、达到论文全部结果、未见动作泛化、多seed鲁棒性、千倍加速、零样本实机迁移、已完成Dex5方法性能提升。

对于“优化训练速度”“提升成功率”等因果表述，当前没有对应消融对照，不应使用。2小时是源码累计训练计时，不是完整租机耗时。专门compiled-BPTT一致性测试超时，不能写成全部梯度验证通过。

## 与Dex5主项目的关系

这段项目适合作为“世界模型与模型驱动策略优化的研究基础”。Dex5简历主线应在实际实现并验证扩展后，写清自己的改进、对照基线与实测收益。当前Dex5仅完成局部仿真诊断：20关节方向和四元数检查通过，10秒候选持笔通过，但60秒检查存在慢滑移；没有转笔策略性能提升结果。
