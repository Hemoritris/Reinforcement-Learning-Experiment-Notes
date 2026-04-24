# Experiment 6: Deep Q-Network (DQN) on Atari Breakout

This experiment focuses on deep reinforcement learning by implementing a classic Deep Q-Network (DQN) agent to solve the Atari `ALE/Breakout-v5` environment. The project covers the full workflow of Atari preprocessing, replay-buffer-based training, checkpoint saving, and post-training policy visualization.

## 📘 Experiment Overview

1. **Environment Construction**: Use the Atari `ALE/Breakout-v5` environment and apply standard DQN preprocessing, including grayscale conversion, frame skipping, screen resizing to `84×84`, and 4-frame stacking.
2. **Algorithm Implementation**:
   - **Deep Q-Network**: Use a convolutional neural network to estimate action values from stacked image observations.
   - **Experience Replay**: Store transitions in a replay buffer and sample mini-batches for stable updates.
   - **Target Network**: Periodically synchronize a target Q-network to reduce training oscillation.
   - **Exploration Strategy**: Adopt an $\epsilon$-greedy policy with linear decay from exploration to exploitation.
3. **Training and Evaluation**: Save checkpoints during training, record average returns, exploration rate, and samples-per-second (SPS), then load trained weights for visualization and video recording.

## 📂 Code Files

| File Name | Description |
|--------|------|
| `Code/train_breakout_dqn.py` | DQN training script for Breakout, including Atari preprocessing, replay buffer, target network update, checkpoint saving, and training logs |
| `Code/visualize_breakout_dqn.py` | Loads a trained checkpoint for evaluation, renders gameplay, and optionally records videos |
| `Code/checkpoints/` | Saved model checkpoints at different training steps |
| `Report/Experiment_6_Report.pdf` | Detailed experiment report |
| `Report/avg_return_20.png` | Training curve of recent average return |
| `Report/eps.png` | Exploration-rate decay curve |
| `Report/SPS.png` | Training throughput curve |
| `Report/best.mp4` | Recorded gameplay video of a trained policy |

## 📄 Experiment Report

Please refer to `Report/Experiment_6_Report.pdf` in this directory for the detailed experiment report and result analysis.

## 🧠 Core Algorithm Formulas

- **DQN Target**:
$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

- **Q-Value Update Objective**:
$$Q_{\theta}(s, a) \leftarrow Q_{\theta}(s, a) + \alpha \left[ y - Q_{\theta}(s, a) \right]$$

- **Training Loss**:
$$L(\theta) = \mathbb{E}\left[\ell_{\text{Huber}}\left(Q_{\theta}(s, a), y\right)\right]$$

## ⚙️ Experiment Parameters

| Parameter | Value |
|------|-----|
| Environment | `ALE/Breakout-v5` |
| Total Training Steps | 1,000,000 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Discount Factor $\gamma$ | 0.99 |
| Replay Buffer Size | 20,000 |
| Learning Starts | 5,000 |
| Train Frequency | Every 4 steps |
| Target Update Interval | 10,000 steps |
| Exploration Rate $\epsilon$ | 1.0 → 0.1 |
| Exploration Decay Steps | 1,000,000 |
| Checkpoint Save Interval | 100,000 steps |

## 📚 References

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Gymnasium Documentation: Atari Environments and Wrappers.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.

## 🗓️ Update Date

April 25, 2026
