from __future__ import annotations  # 延迟解析类型标注，保持运行时轻量。

import argparse  # 读取命令行超参数。
import json  # 保存可读的实验指标。
import math  # 计算上游源码使用的指数 epsilon 衰减。
from collections import defaultdict  # 按上游代码延迟创建状态动作值。
from pathlib import Path  # 构造跨平台的产物路径。

import gymnasium as gym  # 提供 CliffWalking 环境。
import numpy as np  # 存储并更新 Q 表。


class QLearningAgent:  # 只保存离散状态-动作价值的表格智能体。
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, seed: int) -> None:
        self.n_actions = n_actions  # 上游代码保存离散动作总数供随机探索使用。
        self.q_table = defaultdict(lambda: np.zeros(n_actions))  # 上游实现以状态字符串映射到动作价值向量。
        self.learning_rate = learning_rate  # α，单次 TD 更新的步长。
        self.gamma = gamma  # γ，未来回报的折扣因子。
        self.sample_count = 0  # 上游实现用全局采样次数驱动 epsilon。
        self.epsilon_start = 0.95  # 书中配套代码的初始探索率。
        self.epsilon_end = 0.01  # 书中配套代码的最终探索率。
        self.epsilon_decay = 300  # 书中配套代码的指数衰减尺度。

    def select_action(self, state: int, epsilon: float) -> int:
        self.sample_count += 1  # 对齐上游的每一步 epsilon 更新。
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.sample_count / self.epsilon_decay)  # 上游指数 epsilon 衰减。
        if np.random.uniform(0, 1) > epsilon:  # 上游以 1-epsilon 概率选择贪婪动作。
            return int(np.argmax(self.q_table[str(state)]))  # 读取该状态的最大 Q 动作。
        return int(np.random.choice(self.n_actions))  # 否则在离散动作间随机探索。

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        q_predict = self.q_table[str(state)][action]  # 读取上游表结构中的 Q(s,a)。
        q_target = reward if done else reward + self.gamma * np.max(self.q_table[str(next_state)])  # 上游 Bellman 最优目标。
        self.q_table[str(state)][action] += self.learning_rate * (q_target - q_predict)  # 上游 Q-learning 更新。

    def predict_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[str(state)]))  # 上游测试阶段始终选择贪婪动作。


def evaluate(env: gym.Env, agent: QLearningAgent, episodes: int) -> float:
    returns = []  # 记录每个纯贪婪回合的总回报。
    for _ in range(episodes):  # 多回合平均以减少单次轨迹偶然性。
        state, _ = env.reset()  # 取得离散初始状态编号。
        done = False  # 同时覆盖 terminated 与 truncated。
        episode_return = 0.0  # G_0 = Σ_t r_t。
        while not done:
            action = agent.predict_action(state)  # 对齐上游 test 函数的纯贪婪动作。
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
    parser.add_argument("--episodes", type=int, default=400)  # 上游配置的训练回合数。
    parser.add_argument("--learning-rate", type=float, default=0.1)  # 上游学习率 α。
    parser.add_argument("--gamma", type=float, default=0.9)  # 上游折扣因子 γ。
    parser.add_argument("--epsilon-start", type=float, default=1.0)  # 初始完全探索。
    parser.add_argument("--epsilon-end", type=float, default=0.05)  # 末尾保留少量探索。
    parser.add_argument("--seed", type=int, default=42)  # 全实验随机种子。
    args = parser.parse_args()  # 将命令行文本解析成具名参数。

    agent, returns, evaluation_return = train(**vars(args))  # 执行完整训练与贪婪评估。
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"  # 产物与源码隔离。
    artifact_dir.mkdir(exist_ok=True)  # 首次训练时创建目录。
    q_values = np.stack([agent.q_table[str(state)] for state in range(48)])  # 将上游字典 Q 表导出为演示可读取的 (48, 4) 数组。
    np.save(artifact_dir / "q_table.npy", q_values)  # 供 demo.py 复用已学好的策略。
    (artifact_dir / "results.json").write_text(
        json.dumps(
            {"q_table_shape": list(q_values.shape), "final_train_return": returns[-1], "mean_eval_return": evaluation_return, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",  # 明确写成 UTF-8，便于跨平台读取。
    )
    print(f"Q-table shape: {q_values.shape}")  # 验证导出的状态-动作表维度。
    print(f"Final training return: {returns[-1]:.1f}")  # 最后一回合含探索，波动较大。
    print(f"Mean greedy evaluation return (100 episodes): {evaluation_return:.2f}")  # 应接近安全最短路径的 -13。


if __name__ == "__main__":
    main()  # 仅在直接运行本文件时启动训练。
