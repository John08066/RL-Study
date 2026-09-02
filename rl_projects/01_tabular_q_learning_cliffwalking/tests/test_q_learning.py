import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import QLearningAgent


class QLearningAgentTest(unittest.TestCase):
    def test_terminal_transition_does_not_bootstrap(self) -> None:
        agent = QLearningAgent(2, 2, learning_rate=1.0, gamma=0.99, seed=0)
        agent.q_table[1] = [100.0, 100.0]

        agent.update(state=0, action=0, reward=-1.0, next_state=1, done=True)

        self.assertEqual(agent.q_table[0, 0], -1.0)


if __name__ == "__main__":
    unittest.main()
