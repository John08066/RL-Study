# 上游源码基线

- 仓库：https://github.com/datawhalechina/easy-rl
- 固定提交：6b7df8451f74f16d5efb6abc1b94a8746890a0ad
- 原始文件：notebooks/Q-learning/QLearning.ipynb
- 算法主体：QLearning、sample_action、predict_action、update、训练与测试循环。

本项目仅为当前 Gymnasium API 补齐 reset/step 返回值适配、行尾学习注释、终端演示和单元测试；不改变上游 Q-learning 更新式、指数 epsilon 衰减或原始超参数。
