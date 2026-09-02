import sys  # 将源码目录加入模块搜索路径。
import unittest  # 使用标准库测试框架。
from pathlib import Path  # 根据测试位置找到 src。

import torch  # 构造测试张量。

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 无需安装包即可导入训练模块。

from train import QNetwork, td_targets  # 测试网络输出和 TD 目标。


class DQNTest(unittest.TestCase):  # 覆盖两个关键张量契约。
    def test_network_outputs_one_value_per_action(self) -> None:
        values = QNetwork(n_states=4, n_actions=2)(torch.zeros((3, 4)))  # 三个状态构成形状 (3, 4) 的输入。
        self.assertEqual(tuple(values.shape), (3, 2))  # 每个状态必须输出两个动作价值。

    def test_terminal_target_does_not_bootstrap(self) -> None:
        targets = td_targets(torch.tensor([100.0, 100.0]), torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 0.0]), gamma=0.99)  # 第一条终止，第二条不终止。
        self.assertEqual(targets[0].item(), -1.0)  # 终止转移的目标只能是即时奖励。
        self.assertAlmostEqual(targets[1].item(), 98.0)  # 非终止转移应保留未来价值。


if __name__ == "__main__":
    unittest.main()  # 支持直接运行测试文件。
