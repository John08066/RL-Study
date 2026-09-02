import sys  # 将源码目录加入测试进程的模块搜索路径。
import unittest  # 使用 Python 标准库测试框架。
from pathlib import Path  # 从当前测试文件定位项目目录。

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 无需安装包即可导入 train.py。

from train import QLearningAgent  # 只测试 Q 更新的核心行为。


class QLearningAgentTest(unittest.TestCase):  # 验证 Q-learning 更新的关键边界条件。
    def test_terminal_transition_does_not_bootstrap(self) -> None:
        agent = QLearningAgent(2, 2, learning_rate=1.0, gamma=0.99, seed=0)  # 用 α=1 使更新结果可精确断言。
        agent.q_table[str(1)] = [100.0, 100.0]  # 若错误 bootstrap，此值会污染终止状态的更新。

        agent.update(state=0, action=0, reward=-1.0, next_state=1, done=True)  # 终止转移的 TD 目标只能是即时奖励 -1。

        self.assertEqual(agent.q_table[str(0)][0], -1.0)  # 确认没有添加 γ max_a Q(s', a)。


if __name__ == "__main__":
    unittest.main()  # 支持直接执行本测试文件。
