# 02 - DQN：CartPole

本项目对应《Easy RL：强化学习教程》第 7.8 节“使用深度 Q 网络控制 CartPole-v0”。原书中的 CartPole-v0 每回合最多 200 步；当前 Gymnasium 使用兼容的 CartPole-v1，但代码显式将每回合限制为 200 步，因此状态维度、动作空间和任务上限都与书中案例对齐。

与 01_tabular_q_learning_cliffwalking 的核心区别是：这里的连续状态 s ∈ R^4 无法建立有限 Q 表，所以使用网络 Q_θ(s, a) 近似两个动作的价值。训练仍使用 Q-learning 目标，但增加了经验回放与目标网络来稳定更新。

训练奖励、MLP 隐藏层维度、MSE 损失、回放容量、指数 epsilon 衰减、训练回合数和目标网络更新频率均采用上游 notebook 默认值。唯一算法修正见 UPSTREAM_SOURCE.md。

## 独立环境与运行

在仓库根目录按顺序执行：

~~~powershell
./.venv/Scripts/python.exe -m venv ./rl_projects/02_dqn_cartpole/.venv
./rl_projects/02_dqn_cartpole/.venv/Scripts/python.exe -m pip install -r ./rl_projects/02_dqn_cartpole/requirements.txt
./rl_projects/02_dqn_cartpole/.venv/Scripts/python.exe ./rl_projects/02_dqn_cartpole/src/train.py --episodes 200
./rl_projects/02_dqn_cartpole/.venv/Scripts/python.exe -m unittest discover -s ./rl_projects/02_dqn_cartpole/tests -v
./rl_projects/02_dqn_cartpole/.venv/Scripts/python.exe ./rl_projects/02_dqn_cartpole/src/demo.py
~~~

训练保存 artifacts/dqn_cartpole.pt 与 artifacts/results.json；它们属于实验产物，不提交 Git。全程在 CPU 可运行，不需要数据集或 4090。

## 如何观察实际效果

demo.py 加载训练权重，以纯贪婪策略控制 CartPole，并把环境状态绘制为 artifacts/cartpole_demo.gif。输出中的 return 是保持平衡的步数；接近 200 表示已完成书中的 CartPole-v0 难度。若 200 回合的上游默认配置未收敛，应记录结果后再增加回合数，而不改变上游超参数。

## 从 Q-learning 到 DQN

| 表格 Q-learning | DQN |
| --- | --- |
| 参数：Q[s, a]，形状 (48, 4) | 参数：神经网络权重 θ，输入 (batch, 4)，输出 (batch, 2) |
| 更新单个表格元素 | 用一批回放样本最小化 TD 误差 |
| max_a' Q(s', a') 直接查表 | 由冻结的目标网络计算 |
| 适合有限离散状态 | 可处理连续观测，但不自动适合连续动作 |

下一项应是独立的 03_ddpg_pendulum，对应书中 10.4 节；不要用 DQN 直接处理 Pendulum 的连续力矩。
