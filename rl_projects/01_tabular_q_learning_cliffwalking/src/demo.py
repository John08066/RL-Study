import argparse  # 读取是否逐步暂停的选项。
from pathlib import Path  # 定位训练保存的 Q 表。

import gymnasium as gym  # 创建与训练相同的 CliffWalking 环境。
import numpy as np  # 加载 Q 表并选择动作。

ROWS, COLUMNS = 4, 12  # CliffWalking-v0 的固定网格尺寸。
START_STATE, GOAL_STATE = 36, 47  # 左下角起点与右下角目标的状态编号。
CLIFF_STATES = set(range(37, 47))  # 起点和目标之间的十个悬崖格。
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")  # Gymnasium 的四个离散动作编号含义。


def render(state: int) -> str:
    cells = ["." for _ in range(ROWS * COLUMNS)]  # 先将所有普通网格显示为点。
    for cliff_state in CLIFF_STATES:
        cells[cliff_state] = "C"  # 悬崖格显示为 C。
    cells[GOAL_STATE] = "G"  # 固定显示目标位置。
    cells[state] = "A"  # 用 A 覆盖智能体当前位置。
    return "\n".join(" ".join(cells[row * COLUMNS : (row + 1) * COLUMNS]) for row in range(ROWS))  # 按四行格式化网格。


def greedy_action(q_table: np.ndarray, state: int) -> int:
    return int(np.flatnonzero(q_table[state] == q_table[state].max())[0])  # 固定选择第一个最大值，使演示可复现。


def main() -> None:
    parser = argparse.ArgumentParser()  # 构建命令行接口。
    parser.add_argument("--pause", action="store_true")  # 每一步等待 Enter，便于观察轨迹。
    args = parser.parse_args()  # 解析演示选项。

    artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"  # 训练产物目录。
    q_table_path = artifact_dir / "q_table.npy"  # 已训练策略的文件路径。
    if not q_table_path.exists():
        raise FileNotFoundError("未找到 artifacts/q_table.npy；请先运行 train.py。")  # 防止在未训练时误演示全零 Q 表。
    q_table = np.load(q_table_path)  # 读取形状为 (48, 4) 的价值表。

    env = gym.make("CliffWalking-v0")  # 创建演示环境。
    state, _ = env.reset(seed=42)  # 从确定的起点开始。
    total_reward = 0.0  # 累积演示轨迹回报。
    print("A=agent, C=cliff, G=goal, .=safe cell")  # 说明网格符号。
    print(render(state))  # 显示起始网格。

    for step in range(30):  # 安全上限，最优路径只需要 13 步。
        action = greedy_action(q_table, state)  # 根据当前状态的最大 Q 值选择动作。
        next_state, reward, terminated, truncated, _ = env.step(action)  # 在环境中实际执行该动作。
        total_reward += reward  # 累积每一步的 -1 或悬崖惩罚。
        print(f"\nStep {step + 1}: {ACTION_NAMES[action]}, reward={reward:.0f}")  # 报告动作与即时奖励。
        print(render(next_state))  # 显示动作后的新位置。
        if args.pause:
            input("Press Enter for the next step...")  # 允许手动逐帧观看。
        state = next_state  # 推进到下一个决策状态。
        if terminated or truncated:
            break  # 到达目标或环境截断后结束演示。

    env.close()  # 释放环境资源。
    print(f"\nFinished: return={total_reward:.0f}, reached_goal={state == GOAL_STATE}")  # 汇总是否走到目标。


if __name__ == "__main__":
    main()  # 仅在直接运行本文件时启动演示。
