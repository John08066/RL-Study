# 上游源码基线

- 仓库：https://github.com/datawhalechina/easy-rl
- 固定提交：6b7df8451f74f16d5efb6abc1b94a8746890a0ad
- 原始文件：notebooks/DQN.ipynb
- 算法主体：MLP、ReplayBuffer、DQN、训练与测试循环。

本项目仅为当前 Gymnasium API 补齐 reset/step 返回值适配、行尾学习注释、GIF 演示和单元测试。上游 policy_net = model 与 target_net = model 会使两个网络引用同一对象，无法形成目标网络；此处使用 copy.deepcopy(model) 修正为上游代码意图所描述的独立目标网络。除此之外，不保留旧项目的终止奖励塑形或自定义默认超参数。
