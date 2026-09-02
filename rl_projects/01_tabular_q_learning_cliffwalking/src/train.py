from __future__ import annotations  # 延迟解析类型标注，保持运行时轻量。

import argparse  # 读取命令行超参数。
import json  # 保存可读的实验指标。
from pathlib import Path  # 构造跨平台的产物路径。

import gymnasium as gym  # 提供 CliffWalking 环境。
import numpy as np  # 存储并更新 Q 表。


class QLearningAgent:  # 只保存离散状态-动作价值的表格智能体。
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, seed: int) -> None:
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)  # Q[s, a] 的形状为 (48, 4)。
        self.learning_rate = learning_rate  # α，单次 TD 更新的步长。
        self.gamma = gamma  # γ，未来回报的折扣因子。
        self.rng = np.random.default_rng(seed)  # 独立随机数发生器，便于复现探索。

    def select_action(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:  # 以 ε 概率探索而非利用当前价值。
            return int(self.rng.integers(self.q_table.shape[1]))  # 在四个离散动作中均匀随机选择。
        return int(self.rng.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max())))  # 贪婪地选最大 Q，平局时随机打破。

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        bootstrap = 0.0 if done else self.gamma * self.q_table[next_state].max()  # 终止状态不估计未来价值。
        td_error = reward + bootstrap - self.q_table[state, action]  # δ = TD 目标 - 当前估计。
        self.q_table[state, action] += self.learning_rate * td_error  # Q-learning 的 Bellman 最优备份。


def evaluate(env: gym.Env, agent: QLearningAgent, episodes: int) -> float:
    returns = []  # 记录每个纯贪婪回合的总回报。
    for _ in range(episodes):  # 多回合平均以减少单次轨迹偶然性。
        state, _ = env.reset()  # 取得离散初始状态编号。
        done = False  # 同时覆盖 terminated 与 truncated。
        episode_return = 0.0  # G_0 = Σ_t r_t。
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # 评估时不再随机探索。
            state, reward, terminated, truncated, _ = env.step(action)  # 执行动作并获得 s'、r。
            done = terminated or truncated  # Gymnasium 将正常终止与时间截断分开返回。
            episode_return += reward  # 累加当前轨迹奖励。
        returns.append(episode_return)  # 保存本回合回报。
    return float(np.mean(returns))  # 返回 100 回合的平均性能。


def train(episodes: int, learning_rate: float, gamma: float, epsilon_start: float, epsilon_end: float, seed: int) -> tuple[QLearningAgent, list[float], float]:
    env = gym.make("CliffWalking-v0")  # 创建 4×12 的离散悬崖环境。
    n_states = env.observation_space.n  # 48 个网格状态。
    n_actions = env.action_space.n  # 上、右、下、左四个动作。
    agent = QLearningAgent(n_states, n_actions, learning_rate, gamma, seed)  # 初始化所有 Q 值为 0。
    returns = []  # 记录训练期每个回合的回报。

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)  # 固定而不同的每回合随机种子。
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / episodes)  # 线性衰减探索率。
        done = False  # 回合尚未终止。
        episode_return = 0.0  # 统计训练轨迹而非用于更新。
        while not done:
            action = agent.select_action(state, epsilon)  # ε-greedy 采样行为动作 a。
            next_state, reward, terminated, truncated, _ = env.step(action)  # 获得一次真实转移 (s, a, r, s')。
            done = terminated or truncated  # 标记 bootstrap 是否应关闭。
            agent.update(state, action, reward, next_state, done)  # 用 max_a' Q(s', a') 更新 Q(s, a)。
            state = next_state  # 推进到下一时刻状态。
            episode_return += reward  # 累积本训练回合回报。
        returns.append(episode_return)  # 保存学习曲线的一个点。

    evaluation_return = evaluate(env, agent, episodes=100)  # 在无探索条件下衡量学到的策略。
    env.close()  # 释放环境资源。
    return agent, returns, evaluation_return  # 将模型、训练曲线和评估指标交给调用方。


def main() -> None:
    parser = argparse.ArgumentParser()  # 允许从终端覆盖默认超参数。
    parser.add_argument("--episodes", type=int, default=1000)  # 训练回合数。
    parser.add_argument("--learning-rate", type=float, default=0.5)  # α。
    parser.add_argument("--gamma", type=float, default=0.99)  # γ。
    parser.add_argument("--epsilon-start", type=float, default=1.0)  # 初始完全探索。
    parser.add_argument("--epsilon-end", type=float, default=0.05)  # 末尾保留少量探索。
    parser.add_argument("--seed", type=int, default=42)  # 全实验随机种子。
    args = parser.parse_args()  # 将命令行文本解析成具名参数。

    agent, returns, evaluation_return = train(**vars(args))  # 执行完整训练与贪婪评估。
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"  # 产物与源码隔离。
    artifact_dir.mkdir(exist_ok=True)  # 首次训练时创建目录。
    np.save(artifact_dir / "q_table.npy", agent.q_table)  # 供 demo.py 复用已学好的策略。
    (artifact_dir / "results.json").write_text(
        json.dumps(
            {"q_table_shape": list(agent.q_table.shape), "final_train_return": returns[-1], "mean_eval_return": evaluation_return, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",  # 明确写成 UTF-8，便于跨平台读取。
    )
    print(f"Q-table shape: {agent.q_table.shape}")  # 验证状态-动作表的维度。
    print(f"Final training return: {returns[-1]:.1f}")  # 最后一回合含探索，波动较大。
    print(f"Mean greedy evaluation return (100 episodes): {evaluation_return:.2f}")  # 应接近安全最短路径的 -13。


if __name__ == "__main__":
    main()  # 仅在直接运行本文件时启动训练。
