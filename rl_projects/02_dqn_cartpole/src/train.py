from __future__ import annotations  # 延迟解析类型标注。

import argparse  # 读取命令行超参数。
import copy  # 为上游目标网络创建独立参数副本。
import json  # 保存实验指标。
import math  # 计算上游的指数 epsilon 衰减。
import random  # 固定回放采样的随机性。
from collections import deque  # 实现固定容量经验池。
from dataclasses import dataclass  # 表示一条环境转移。
from pathlib import Path  # 定位实验产物。

import gymnasium as gym  # 提供 CartPole-v1。
import numpy as np  # 处理连续状态批次。
import torch  # 训练并保存神经网络。
from torch import nn  # 定义网络和损失。


@dataclass
class Transition:  # 一条 (s, a, r, s', done) 经验。
    state: np.ndarray  # 当前连续状态，形状 (4,)。
    action: int  # 左或右动作，取值 0/1。
    reward: float  # 即时奖励。
    next_state: np.ndarray  # 下一连续状态，形状 (4,)。
    done: bool  # 终止时不能 bootstrap。


class ReplayBuffer:  # 打乱时间相邻样本，稳定梯度更新。
    def __init__(self, capacity: int, seed: int) -> None:
        self.items: deque[Transition] = deque(maxlen=capacity)  # 满时自动移除最旧经验。
        self.rng = random.Random(seed)  # 独立随机源，便于复现。

    def append(self, item: Transition) -> None:
        self.items.append(item)  # 保存一次真实环境交互。

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.items, batch_size)  # 无放回抽样训练批次。

    def __len__(self) -> int:
        return len(self.items)  # 支持判断预热是否结束。


class QNetwork(nn.Module):  # 近似 Q_theta(s, a)，输出两个动作价值。
    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()  # 初始化 PyTorch 基类。
        self.layers = nn.Sequential(  # 输入 (batch, 4)，输出 (batch, 2)。
            nn.Linear(n_states, 256),  # 上游 MLP 的第一隐藏层维度。
            nn.ReLU(),  # 增加非线性表达能力。
            nn.Linear(256, 256),  # 上游 MLP 的第二隐藏层维度。
            nn.ReLU(),  # 第二次非线性变换。
            nn.Linear(256, n_actions),  # 上游 MLP 为每个离散动作输出 Q 值。
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)  # 保持 batch 维并返回动作价值。


def td_targets(next_q: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    return rewards + gamma * (1.0 - dones) * next_q  # done=1 时未来价值严格为零。


class DQNAgent:  # 维护在线网络、目标网络和一次梯度更新。
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, seed: int) -> None:
        self.online_net = QNetwork(n_states, n_actions)  # 被优化的 Q_theta。
        self.target_net = copy.deepcopy(self.online_net)  # 修复上游同一模型引用，使目标网络独立但初始相同。
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)  # 更新在线网络参数。
        self.loss_fn = nn.MSELoss()  # 对齐上游 DQN.ipynb 的 MSE TD 损失。
        self.gamma = gamma  # 折扣因子。
        self.n_actions = n_actions  # 随机探索时需要的动作数。
        self.rng = np.random.default_rng(seed)  # ε-greedy 的随机数发生器。

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:  # 以 ε 概率探索。
            return int(self.rng.integers(self.n_actions))  # 在两个动作间随机选择。
        with torch.no_grad():  # 推理时不构建反向传播图。
            values = self.online_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))  # 将 (4,) 扩为 (1, 4)。
        return int(values.argmax(dim=1).item())  # 利用时选最大 Q 值动作。

    def update(self, batch: list[Transition]) -> float:
        states = torch.as_tensor(np.stack([x.state for x in batch]), dtype=torch.float32)  # 状态批次形状 (batch, 4)。
        actions = torch.as_tensor([x.action for x in batch], dtype=torch.int64).unsqueeze(1)  # 动作索引形状 (batch, 1)。
        rewards = torch.as_tensor([x.reward for x in batch], dtype=torch.float32)  # 奖励形状 (batch,)。
        next_states = torch.as_tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32)  # 下一状态批次。
        dones = torch.as_tensor([x.done for x in batch], dtype=torch.float32)  # bool 转换为 0/1。
        predicted_q = self.online_net(states).gather(1, actions).squeeze(1)  # 取 Q_theta(s, 实际动作)。
        with torch.no_grad():  # 目标网络不参与反向传播。
            next_q = self.target_net(next_states).max(dim=1).values  # max_a' Q_theta_minus(s', a')。
            target_q = td_targets(next_q, rewards, dones, self.gamma)  # 形成 Bellman 最优目标。
        loss = self.loss_fn(predicted_q, target_q)  # 计算 TD 损失。
        self.optimizer.zero_grad()  # 清除上一批梯度。
        loss.backward()  # 反向传播到在线网络。
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 100.0)  # 限制不稳定的大梯度。
        self.optimizer.step()  # 执行参数更新。
        return float(loss.item())  # 返回标量损失供观察。

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())  # 周期性更新目标网络，稳定学习目标。


def evaluate(agent: DQNAgent, episodes: int, seed: int) -> float:
    env = gym.make("CartPole-v1", max_episode_steps=200)  # 使用 v1 API，同时复现书中 v0 的 200 步上限。
    returns = []  # 收集贪婪策略的回合回报。
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)  # 固定评估初态序列。
        done = False  # 同时覆盖 terminated 与 truncated。
        total_reward = 0.0  # CartPole 每个存活步骤奖励为 1。
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # 评估时取消探索。
            state, reward, terminated, truncated, _ = env.step(action)  # 执行贪婪策略。
            done = terminated or truncated  # 失败或步数上限结束回合。
            total_reward += reward  # 累积回合奖励。
        returns.append(total_reward)  # 保存本回合表现。
    env.close()  # 释放环境。
    return float(np.mean(returns))  # 返回多回合平均值。


def train(episodes: int, learning_rate: float, gamma: float, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int, batch_size: int, buffer_size: int, warmup_steps: int, target_sync_steps: int, seed: int) -> tuple[DQNAgent, list[float], float]:
    random.seed(seed)  # 固定 Python 随机数。
    np.random.seed(seed)  # 固定 NumPy 随机数。
    torch.manual_seed(seed)  # 固定网络初始化。
    env = gym.make("CartPole-v1", max_episode_steps=200)  # 使用 v1 API，同时复现书中 v0 的 200 步上限。
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, seed)  # 状态维 4，动作数 2。
    replay = ReplayBuffer(buffer_size, seed)  # 创建经验回放池。
    returns = []  # 记录训练期回报。
    total_steps = 0  # 驱动 ε 衰减和目标同步。

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)  # 每回合使用可复现但不同的初态。
        done = False  # 初始化终止标志。
        episode_return = 0.0  # 初始化训练回报。
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-total_steps / epsilon_decay_steps)  # 对齐上游的指数 epsilon 衰减。
            action = agent.select_action(state, epsilon)  # ε-greedy 行为策略。
            next_state, reward, terminated, truncated, _ = env.step(action)  # 获得真实转移。
            done = terminated or truncated  # 合并 Gymnasium 的两类结束。
            replay.append(Transition(state, action, reward, next_state, done))  # 对齐上游，直接保存环境原始奖励。
            state = next_state  # 推进状态。
            episode_return += reward  # 累积本回合奖励。
            total_steps += 1  # 增加全局环境步数。
            if len(replay) >= max(batch_size, warmup_steps):  # 预热后再开始批量学习。
                agent.update(replay.sample(batch_size))  # 随机回放历史经验。
        returns.append(episode_return)  # 保存训练曲线点。
        if (episode + 1) % target_sync_steps == 0:  # 对齐上游，按训练回合同步目标网络。
            agent.sync_target()  # 复制在线网络参数。

    score = evaluate(agent, episodes=30, seed=seed + episodes)  # 用纯贪婪策略评估。
    env.close()  # 释放训练环境。
    return agent, returns, score  # 返回模型、曲线和评估值。


def main() -> None:
    parser = argparse.ArgumentParser()  # 创建命令行接口。
    parser.add_argument("--episodes", type=int, default=200)  # 上游配置的训练回合数。
    parser.add_argument("--learning-rate", type=float, default=1e-4)  # 上游 Adam 学习率。
    parser.add_argument("--gamma", type=float, default=0.95)  # 上游折扣因子。
    parser.add_argument("--epsilon-start", type=float, default=0.95)  # 上游初始探索率。
    parser.add_argument("--epsilon-end", type=float, default=0.01)  # 上游最终探索率。
    parser.add_argument("--epsilon-decay-steps", type=int, default=500)  # 上游指数衰减尺度。
    parser.add_argument("--batch-size", type=int, default=64)  # 每次更新的样本数。
    parser.add_argument("--buffer-size", type=int, default=100_000)  # 上游回放池容量。
    parser.add_argument("--warmup-steps", type=int, default=1_000)  # 首次更新前收集的经验数。
    parser.add_argument("--target-sync-steps", type=int, default=4)  # 上游按回合进行的目标网络同步间隔。
    parser.add_argument("--seed", type=int, default=42)  # 实验随机种子。
    args = parser.parse_args()  # 解析参数。

    agent, returns, score = train(**vars(args))  # 执行训练。
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"  # 与源码分隔实验产物。
    artifact_dir.mkdir(exist_ok=True)  # 首次训练时创建目录。
    torch.save(agent.online_net.state_dict(), artifact_dir / "dqn_cartpole.pt")  # 保存在线网络权重。
    (artifact_dir / "results.json").write_text(json.dumps({"final_train_return": returns[-1], "mean_eval_return": score, "seed": args.seed}, indent=2), encoding="utf-8")  # 保存关键指标。
    print(f"Final training return: {returns[-1]:.1f}")  # 最后一回合仍有探索噪声。
    print(f"Mean greedy evaluation return (30 episodes): {score:.2f}")  # 主要学习效果指标。


if __name__ == "__main__":
    main()  # 直接运行文件时启动训练。
