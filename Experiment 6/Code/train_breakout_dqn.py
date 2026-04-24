#!/usr/bin/env python3
import argparse
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.error import NamespaceNotFound


class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # 经典 Atari DQN 卷积骨干：输入 4 帧堆叠图像，输出每个离散动作的 Q 值。
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # 用 uint8 存图像可显著降低内存占用，训练时再归一化到 [0,1]。
        self.states = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_states = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


def set_seed(seed: int, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, render_mode: str | None = None):
    # Gymnasium Atari 环境需要先注册 ALE。
    try:
        import ale_py  # type: ignore

        gym.register_envs(ale_py)
    except Exception:
        pass

    make_kwargs = {
        # 与训练/评估保持一致的 ALE 配置，避免环境漂移。
        "frameskip": 1,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "mode": 0,
        "difficulty": 0,
    }
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(env_id, **make_kwargs)
    except TypeError:
        # Backward-compatible fallback for older ALE/Gymnasium APIs.
        make_kwargs.pop("full_action_space", None)
        make_kwargs.pop("mode", None)
        make_kwargs.pop("difficulty", None)
        env = gym.make(env_id, **make_kwargs)
    except NamespaceNotFound as exc:
        raise RuntimeError(
            "Atari namespace is not available. Install Atari support with:\n"
            "  pip install gymnasium[atari,accept-rom-license] ale-py\n"
            "Then retry training."
        ) from exc

    env = gym.wrappers.AtariPreprocessing(
        env,
        # DQN 常用 Atari 预处理：降采样、灰度、动作重复与最大池化。
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    if hasattr(gym.wrappers, "FrameStackObservation"):
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    else:
        env = gym.wrappers.FrameStack(env, num_stack=4)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def obs_to_chw_uint8(obs) -> np.ndarray:
    x = np.asarray(obs, dtype=np.uint8)
    if x.ndim != 3:
        raise ValueError(f"Unexpected observation shape: {x.shape}")
    # 兼容两种常见布局，并统一为 CHW。
    if x.shape[0] == 4:
        return x
    if x.shape[-1] == 4:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Unexpected observation shape: {x.shape}")


def to_tensor_u8(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    # 从 uint8 转 float32 并归一化。
    return torch.from_numpy(batch).to(device=device, dtype=torch.float32) / 255.0


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    # 线性 epsilon 衰减：前期多探索，后期多利用。
    if step >= decay_steps:
        return eps_end
    ratio = step / max(decay_steps, 1)
    return eps_start + ratio * (eps_end - eps_start)


def save_checkpoint(
    save_dir: str,
    step: int,
    online_net: nn.Module,
    optimizer: optim.Optimizer,
    epsilon: float,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"dqn_breakout_step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model": online_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
        },
        ckpt_path,
    )
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Breakout with CUDA support")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.1)
    parser.add_argument("--eps-decay-steps", type=int, default=1_000_000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--reward-clip", action="store_true", default=True)
    parser.add_argument("--no-reward-clip", dest="reward_clip", action="store_false")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=5_000)
    return parser.parse_args()


def train(args):
    set_seed(args.seed, use_cuda=(args.device == "cuda"))

    # 设备选择：请求 CUDA 但不可用时自动回退到 CPU。
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA is not available, fallback to CPU.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.seed)
    n_actions = env.action_space.n

    online_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    obs, _ = env.reset(seed=args.seed)
    obs = obs_to_chw_uint8(obs)

    episode_return = 0.0
    episode_len = 0
    recent_returns = deque(maxlen=20)

    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)

        # epsilon-greedy：按概率随机探索，否则执行贪心动作。
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = online_net(to_tensor_u8(obs[None, ...], device))
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        next_obs = obs_to_chw_uint8(next_obs)

        # 训练默认用奖励裁剪，通常能提升 Atari 训练稳定性。
        train_reward = np.sign(reward) if args.reward_clip else reward
        replay.add(obs, action, float(train_reward), next_obs, done)

        obs = next_obs
        episode_return += reward
        episode_len += 1

        if done:
            recent_returns.append(episode_return)
            # 回合结束后重置环境，继续收集样本。
            obs, _ = env.reset()
            obs = obs_to_chw_uint8(obs)
            episode_return = 0.0
            episode_len = 0

        if len(replay) >= args.learning_starts and step % args.train_freq == 0:
            for _ in range(args.gradient_steps):
                states, actions, rewards, next_states, dones = replay.sample(args.batch_size)

                states_t = to_tensor_u8(states, device)
                actions_t = torch.from_numpy(actions).to(device=device, dtype=torch.long)
                rewards_t = torch.from_numpy(rewards).to(device=device, dtype=torch.float32)
                next_states_t = to_tensor_u8(next_states, device)
                dones_t = torch.from_numpy(dones).to(device=device, dtype=torch.float32)

                q_values = online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # DQN 目标：r + gamma * max_a' Q_target(s', a')
                    next_q = target_net(next_states_t).max(dim=1).values
                    target = rewards_t + (1.0 - dones_t) * args.gamma * next_q

                loss = F.smooth_l1_loss(q_values, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), args.max_grad_norm)
                optimizer.step()

        if step % args.target_update_interval == 0:
            # 定期同步目标网络，减少训练发散。
            target_net.load_state_dict(online_net.state_dict())

        if step % args.save_interval == 0:
            ckpt = save_checkpoint(args.save_dir, step, online_net, optimizer, epsilon)
            print(f"[Checkpoint] Saved to {ckpt}")

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = int(step / max(elapsed, 1e-6))
            avg_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            print(
                f"step={step} | eps={epsilon:.4f} | buffer={len(replay)} | "
                f"avg_return_20={avg_return:.2f} | SPS={sps} | device={device}"
            )

    final_ckpt = save_checkpoint(args.save_dir, args.total_steps, online_net, optimizer, epsilon)
    print(f"[Done] Final checkpoint: {final_ckpt}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
