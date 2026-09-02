# 01 - Q-learning：CliffWalking

本项目对应《Easy RL：强化学习教程》第 3 章“表格型方法”的 **Q-learning 算法实战**（书中 3.5，CliffWalking-v0）。官方配套仓库为 [datawhalechina/easy-rl](https://github.com/datawhalechina/easy-rl)，该仓库将第 3 章列为 Q-learning 实战；本目录是使用当前 `gymnasium` API 写的最小、可运行实现，不复制整套上游工程。

算法主体现以官方 notebooks/Q-learning/QLearning.ipynb 为准：默认字典 Q 表、指数 epsilon 衰减、学习率 0.1、折扣 0.9、训练 400 回合。当前 Gymnasium API、终端演示和测试属于补充，完整来源记录见 UPSTREAM_SOURCE.md。

## 为什么从这里开始

状态为离散整数 `s ∈ {0, …, 47}`，动作数为 4。因此 `Q` 的形状为 `(48, 4)`，无网络、无数据集、无 GPU。每一步以

`Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]`

更新；若回合终止，则 bootstrap 项为 0。悬崖格会给出 -100 并把智能体送回起点，目标是找出安全的最短路径。

## 环境与运行

在仓库根目录按顺序执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\rl_projects\01_tabular_q_learning_cliffwalking\requirements.txt
.\.venv\Scripts\python.exe .\rl_projects\01_tabular_q_learning_cliffwalking\src\train.py --episodes 1000
.\.venv\Scripts\python.exe -m unittest discover -s .\rl_projects\01_tabular_q_learning_cliffwalking\tests -v
.\.venv\Scripts\python.exe .\rl_projects\01_tabular_q_learning_cliffwalking\src\demo.py --pause
```

首条安装命令假设根目录 `.venv` 已由项目初始化步骤创建。训练会把可复现实验结果写进 `artifacts/`（被 Git 忽略）。本例不需要数据集，也不需要 4090。

## 如何观察它实际运作

1. 运行 `train.py`：它执行 1000 个回合的“探索 → 与环境交互 → Q-learning 更新”，并保存 `artifacts/q_table.npy`。最后的训练回报会受探索影响；重点看 100 回合纯贪婪评估，通常为 -13。
2. 运行测试命令：它验证终止转移不会再加上错误的未来价值。
3. 运行 `demo.py --pause`：每按一次 Enter 执行一个贪婪动作。网格中 `A` 是智能体，`C` 是悬崖，`G` 是目标；最优演示应经过 13 步、总回报为 -13，且 `reached_goal=True`。

## 预期检查点

- 脚本输出 Q 表形状 `(48, 4)`；
- 贪婪评估平均回报应明显优于随机撞悬崖（典型收敛值约为 -13）；
- `tests/test_q_learning.py` 验证终止状态不会错误 bootstrap。

## 下一步

完成后推荐在 `02_dqn_cartpole/` 单独建立 DQN 项目，再转到连续动作的 Pendulum / DDPG。不要把表格 Q-learning 直接等同于具身控制：真实具身任务通常是高维、部分可观测且连续控制，必须更换状态表征、函数逼近器与仿真环境。
