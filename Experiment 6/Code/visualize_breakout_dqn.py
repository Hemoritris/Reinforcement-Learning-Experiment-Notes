#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from train_breakout_dqn import QNetwork, make_env, obs_to_chw_uint8, to_tensor_u8


def set_seed(seed: int, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, num_actions: int, device: torch.device) -> QNetwork:
    # 加载训练好的 Q 网络参数用于评估/演示。
    model = QNetwork(num_actions).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained DQN Breakout policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps-per-episode", type=int, default=4000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--record", action="store_true", default=True)
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.add_argument("--video-dir", type=str, default="videos")
    return parser.parse_args()


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA is not available, fallback to CPU.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed, use_cuda=(device.type == "cuda"))

    env = make_env(args.env_id, args.seed, render_mode=args.render_mode)
    # 打印关键环境配置，便于核对训练与评估的一致性。
    print(
        "env_config: "
        f"env_id={args.env_id}, "
        "ale_mode=0, ale_difficulty=0, "
        "repeat_action_probability=0.0, env_frameskip=1, "
        "preprocess_frame_skip=4, stack_size=4, "
        f"actions={env.unwrapped.get_action_meanings()}"
    )

    if args.record:
        # 可选视频录制，不影响策略推理逻辑。
        os.makedirs(args.video_dir, exist_ok=True)
        name_prefix = f"breakout_{Path(args.checkpoint).stem}"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_dir,
            episode_trigger=lambda _: True,
            video_length=args.max_steps_per_episode,
            name_prefix=name_prefix,
            disable_logger=True,
        )

    model = load_model(args.checkpoint, env.action_space.n, device)

    episode_returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        obs = obs_to_chw_uint8(obs)

        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done and ep_steps < args.max_steps_per_episode:
            # 评估默认 epsilon=0（纯贪心），也可手动提高观察鲁棒性。
            if random.random() < args.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = model(to_tensor_u8(obs[None, ...], device))
                    action = int(q.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            obs = obs_to_chw_uint8(next_obs)

            ep_return += reward
            ep_steps += 1

        episode_returns.append(ep_return)
        print(f"episode={ep + 1} | return={ep_return:.2f} | steps={ep_steps}")

    avg_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
    print(f"avg_return={avg_return:.2f} | episodes={args.episodes} | device={device}")

    if args.record:
        print(f"videos_saved_to={os.path.abspath(args.video_dir)}")

    env.close()


if __name__ == "__main__":
    run(parse_args())
