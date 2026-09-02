"""Tabular Q-learning for Gymnasium's CliffWalking-v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, seed: int) -> None:
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))
        return int(self.rng.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max())))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        bootstrap = 0.0 if done else self.gamma * self.q_table[next_state].max()
        td_error = reward + bootstrap - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error


def evaluate(env: gym.Env, agent: QLearningAgent, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    return float(np.mean(returns))


def train(episodes: int, learning_rate: float, gamma: float, epsilon_start: float, epsilon_end: float, seed: int) -> tuple[QLearningAgent, list[float], float]:
    env = gym.make("CliffWalking-v0")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(n_states, n_actions, learning_rate, gamma, seed)
    returns = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - episode / episodes)
        done = False
        episode_return = 0.0
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
        returns.append(episode_return)

    evaluation_return = evaluate(env, agent, episodes=100)
    env.close()
    return agent, returns, evaluation_return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, returns, evaluation_return = train(**vars(args))
    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    np.save(artifact_dir / "q_table.npy", agent.q_table)
    (artifact_dir / "results.json").write_text(
        json.dumps(
            {"q_table_shape": list(agent.q_table.shape), "final_train_return": returns[-1], "mean_eval_return": evaluation_return, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Q-table shape: {agent.q_table.shape}")
    print(f"Final training return: {returns[-1]:.1f}")
    print(f"Mean greedy evaluation return (100 episodes): {evaluation_return:.2f}")


if __name__ == "__main__":
    main()
