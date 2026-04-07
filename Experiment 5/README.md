# Experiment 5: SARSA and Q-Learning Temporal Difference Algorithms

This experiment focuses on Temporal Difference (TD) reinforcement learning methods. By implementing two classic TD algorithms, SARSA and Q-Learning, to solve the FrozenLake problem, it achieves value function estimation and optimal policy learning, and comparatively analyzes the performance differences between the two algorithms.

## 📝 Experiment Overview

1. **Environment Setup**: Familiarize and use the FrozenLake-v1 environment, mastering its state space, action space, reward mechanism, and environment interaction rules.
2. **Algorithm Implementation**:
   - **SARSA Algorithm**: An on-policy TD control method based on the current policy, using the actual next action executed to update the Q-value.
   - **Q-Learning Algorithm**: An off-policy TD control method based on the optimal policy, using the maximum Q-value of the next state to update.

## 💻 Code Files

| File Name | Description |
|--------|------|
| `SARSA_FrozenLake.py` | Implementation of SARSA algorithm and FrozenLake environment training/testing |
| `QLearning_FrozenLake.py` | Implementation of Q-Learning algorithm and FrozenLake environment training/testing |
| `Compare_SARSA_QLearning.py` | Comparative experiment and visualization of the two algorithms |

## 📄 Experiment Report

Please refer to the PDF document in this directory for the detailed experiment report.

## 🔑 Core Algorithm Formulas

- **SARSA Update Formula**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

- **Q-Learning Update Formula**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

## 📊 Experiment Parameters

| Parameter | Value |
|------|-----|
| Learning Rate $\alpha$ | 0.1 |
| Discount Factor $\gamma$ | 0.99 |
| Initial Exploration Rate $\epsilon$ | 0.5 |
| Minimum Exploration Rate | 0.01 |
| Exploration Decay | 0.9995 |
| Training Episodes | 10000 |

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. (Chapter 6: Temporal-Difference Learning)

## 📅 Update Date

April 2025