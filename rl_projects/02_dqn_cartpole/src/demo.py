import argparse  # 读取演示参数。
from pathlib import Path  # 定位权重和 GIF 文件。

import gymnasium as gym  # 创建 CartPole 物理环境。
import torch  # 加载训练权重并推理。
from PIL import Image, ImageDraw  # 绘制状态帧并编码 GIF。

from train import QNetwork  # 复用训练时的网络结构。


def draw_frame(state: object) -> Image.Image:
    image = Image.new("RGB", (600, 400), "white")  # 创建一帧白色画布。
    draw = ImageDraw.Draw(image)  # 获取 Pillow 绘图接口。
    cart_x = 300.0 + float(state[0]) * 100.0  # 将环境的小车位置映射到像素位置。
    cart_y = 300.0  # 固定轨道高度。
    draw.line((0, cart_y + 20, 600, cart_y + 20), fill="black", width=3)  # 绘制轨道。
    draw.rectangle((cart_x - 40, cart_y - 20, cart_x + 40, cart_y + 20), fill="black")  # 绘制小车。
    pole_x = cart_x + 150.0 * float(torch.sin(torch.tensor(state[2])))  # 根据杆角度计算杆顶端横坐标。
    pole_y = cart_y - 150.0 * float(torch.cos(torch.tensor(state[2])))  # 根据杆角度计算杆顶端纵坐标。
    draw.line((cart_x, cart_y, pole_x, pole_y), fill="red", width=8)  # 绘制真实状态对应的杆。
    return image  # 返回可追加到 GIF 的一帧。


def main() -> None:
    parser = argparse.ArgumentParser()  # 创建命令行接口。
    parser.add_argument("--max-steps", type=int, default=200)  # 与书中的 CartPole-v0 回合上限一致。
    args = parser.parse_args()  # 解析命令行参数。

    project_dir = Path(__file__).resolve().parents[1]  # 定位本项目根目录。
    model_path = project_dir / "artifacts" / "dqn_cartpole.pt"  # 训练权重路径。
    if not model_path.exists():
        raise FileNotFoundError("未找到 artifacts/dqn_cartpole.pt；请先运行 train.py。")  # 禁止未训练时演示全零策略。
    network = QNetwork(n_states=4, n_actions=2)  # 状态维度 4、动作数 2。
    network.load_state_dict(torch.load(model_path, weights_only=True))  # 恢复 Q_theta。
    network.eval()  # 切换推理模式。

    env = gym.make("CartPole-v1", max_episode_steps=200)  # 使用真实动力学并复现书中 200 步上限。
    state, _ = env.reset(seed=42)  # 固定初态使 GIF 可复现。
    frames = [draw_frame(state)]  # 根据起始真实状态绘制一帧。
    total_reward = 0.0  # 累积演示回报。
    done = False  # 初始化终止标志。

    for _ in range(args.max_steps):
        with torch.no_grad():  # 演示阶段不计算梯度。
            values = network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))  # 将状态变为形状 (1, 4)。
        action = int(values.argmax(dim=1).item())  # 选择最大 Q 值动作。
        state, reward, terminated, truncated, _ = env.step(action)  # 在物理环境执行动作。
        frames.append(draw_frame(state))  # 根据真实下一状态绘制该帧。
        total_reward += reward  # 累积保持平衡步数。
        done = terminated or truncated  # 杆倒下或达到书中的 200 步上限则结束。
        if done:
            break  # 停止生成无效后续帧。

    env.close()  # 释放环境资源。
    output_path = project_dir / "artifacts" / "cartpole_demo.gif"  # GIF 输出路径。
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=30, loop=0)  # 写入循环播放的 GIF。
    print(f"Saved demo: {output_path}")  # 报告演示文件位置。
    print(f"Demo return: {total_reward:.0f}, finished={done}")  # 报告实际演示回报。


if __name__ == "__main__":
    main()  # 直接运行文件时生成 GIF。
