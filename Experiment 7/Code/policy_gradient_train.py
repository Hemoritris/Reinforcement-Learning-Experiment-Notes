import argparse
from pathlib import Path
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

try:
    # 优先使用新版 gymnasium；如果没有安装，则退回到旧版 gym。
    import gymnasium as gym
except ImportError:
    import gym


class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出每个离散动作对应的 logits。"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # CartPole 的状态维度较小，两个隐藏层已经足够表达策略。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


def reset_env(env):
    """兼容 gymnasium 和旧版 gym 的 reset 返回格式。"""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    """兼容 gymnasium 和旧版 gym 的 step 返回格式。"""
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info
    next_state, reward, done, info = result
    return next_state, reward, done, info


def discounted_returns(rewards, gamma):
    """计算每个时间步的折扣回报 G_t。"""
    returns = []
    running_return = 0.0
    # 从后往前递推：G_t = r_t + gamma * G_{t+1}
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        # 对单回合回报做标准化，可以降低策略梯度估计的方差。
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def select_action(policy, state, device):
    """根据当前策略采样动作，并返回该动作的 log 概率。"""
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_logits = policy(state_tensor)
    # Categorical(logits=...) 会在内部做 softmax，数值上比手动 softmax 更稳定。
    distribution = Categorical(logits=action_logits)
    action = distribution.sample()
    return action.item(), distribution.log_prob(action)


def save_training_curve(episode_rewards, average_rewards, plot_path):
    """保存训练曲线图，包含单回合奖励和滑动平均奖励。"""
    if not episode_rewards:
        return

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(episode_rewards) + 1)
    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(episodes, episode_rewards, color="#8fb3ff", linewidth=0.9, alpha=0.55, label="Episode reward")
    plt.plot(episodes, average_rewards, color="#d62728", linewidth=1.8, label="Moving average reward")
    plt.axhline(475.0, color="#2ca02c", linestyle="--", linewidth=1.2, label="Solved threshold")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Training Curve on CartPole-v1")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curve to {plot_path}")


def train(args):
    """使用批量 REINFORCE 算法训练策略网络。"""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = gym.make(args.env)

    # 从环境空间中自动读取状态维度和动作数量。
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim, args.hidden_dim).to(device)
    optimizer = Adam(policy.parameters(), lr=args.lr)

    # 保存最近若干回合的奖励，用于判断是否达到解决标准。
    recent_scores = deque(maxlen=args.solved_window)
    episode_rewards = []
    average_rewards = []

    episode = 0
    solved = False

    while episode < args.episodes and not solved:
        # 每 batch 收集多个 episode，再统一做一次策略梯度更新。
        batch_log_probs = []
        batch_returns = []
        batch_losses = []

        for _ in range(args.batch_size):
            if episode >= args.episodes:
                break

            episode += 1
            state = reset_env(env)
            log_probs = []
            rewards = []
            done = False

            # 采样一整条轨迹，REINFORCE 需要完整 episode 的回报。
            while not done:
                action, log_prob = select_action(policy, state, device)
                next_state, reward, done, _ = step_env(env, action)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state

            # 将该回合中每一步的 log_prob 和回报加入 batch。
            returns = discounted_returns(rewards, args.gamma)
            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

            episode_reward = float(np.sum(rewards))
            recent_scores.append(episode_reward)
            mean_score = float(np.mean(recent_scores))
            episode_rewards.append(episode_reward)
            average_rewards.append(mean_score)

            if episode == 1 or episode % args.log_interval == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"reward {episode_reward:7.2f} | "
                    f"{len(recent_scores)}-episode avg {mean_score:7.2f}"
                )

            if len(recent_scores) == args.solved_window and mean_score >= args.solved_score:
                print(
                    f"Solved at episode {episode}: "
                    f"{args.solved_window}-episode average reward = {mean_score:.2f}"
                )
                solved = True
                break

        batch_returns_tensor = torch.stack(batch_returns).to(device)
        if len(batch_returns_tensor) > 1:
            # 再对整个 batch 的回报做一次标准化，使更新更加平稳。
            batch_returns_tensor = (
                batch_returns_tensor - batch_returns_tensor.mean()
            ) / (batch_returns_tensor.std() + 1e-8)
        batch_log_probs_tensor = torch.cat(batch_log_probs)
        # REINFORCE 目标是最大化 log_prob(action) * return；
        # 优化器默认最小化 loss，所以这里取负号。
        batch_losses.append(-(batch_log_probs_tensor * batch_returns_tensor).mean())

        loss = torch.stack(batch_losses).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存环境信息、网络结构参数和训练后的权重，方便之后评估。
    torch.save(
        {
            "env": args.env,
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(args.hidden_dim),
            "model_state_dict": policy.state_dict(),
        },
        args.model_path,
    )
    env.close()
    print(f"Saved model to {args.model_path}")
    save_training_curve(episode_rewards, average_rewards, args.plot_path)


def evaluate(args):
    """加载已保存模型，用贪心策略评估若干回合。"""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # 这里加载的是本脚本刚保存的完整 checkpoint，因此显式允许完整反序列化。
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    render_mode = "human" if args.render else None
    env = gym.make(checkpoint["env"], render_mode=render_mode)

    # 根据 checkpoint 中记录的结构参数重建策略网络。
    policy = PolicyNetwork(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    scores = []
    for episode in range(1, args.eval_episodes + 1):
        state = reset_env(env)
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                # 评估时不再随机采样，直接选择概率最大的动作。
                action = torch.argmax(policy(state_tensor), dim=-1).item()
            state, reward, done, _ = step_env(env, action)
            total_reward += reward

        scores.append(total_reward)
        print(f"Eval episode {episode:2d} | reward {total_reward:7.2f}")

    env.close()
    print(f"Average eval reward: {np.mean(scores):.2f}")


def parse_args():
    """解析命令行参数，便于切换环境、训练轮数和评估模式。"""
    parser = argparse.ArgumentParser(
        description="Train a Gym environment with the REINFORCE policy-gradient algorithm."
    )
    parser.add_argument("--env", default="CartPole-v1", help="Gym environment name.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--solved-score", type=float, default=475.0)
    parser.add_argument("--solved-window", type=int, default=50)
    parser.add_argument("--model-path", default="policy_gradient_cartpole.pth")
    parser.add_argument(
        "--plot-path",
        default="../Report/policy_gradient_training_curve.png",
        help="Path for the saved training curve figure.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA exists.")
    parser.add_argument("--eval", action="store_true", help="Evaluate a saved model.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Render during evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
