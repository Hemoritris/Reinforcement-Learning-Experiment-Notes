# Experiment 4: Monte Carlo Methods & Model-Free RL

This experiment primarily focuses on model-free reinforcement learning methods by implementing classic Monte Carlo (MC) core algorithms to solve the Markov Decision Process (MDP) for the FrozenLake grid world, achieving value function estimation and optimal policy learning.

## 📝 Experiment Overview

1. **Environment Construction**: Familiarize with and use the FrozenLake-v1 grid world environment, mastering the states, actions, rewards, and environment interaction rules in model-free reinforcement learning scenarios.
2. **Algorithm Implementation**:
   - **Monte Carlo with Exploring Starts**: Estimate first-visit Monte Carlo values and optimize policies by randomly initializing state-action pairs.
   - **On-Policy Monte Carlo**: Sample based on the $\epsilon$-greedy policy to synchronously complete policy evaluation and policy improvement.
   - **Off-Policy Monte Carlo**: Combine behavior policy and target policy, utilizing importance sampling to achieve optimal policy learning.
3. **Result Comparison and Analysis**: Compare the convergence speed, training success rate, and policy performance differences between on-policy and off-policy Monte Carlo algorithms, and analyze the impact of exploration vs. exploitation and importance sampling on algorithm effectiveness.

## 💻 Code
- Contains the Python implementation files for the core Monte Carlo algorithms (MC with Exploring Starts, On-Policy MC, Off-Policy MC), as well as the training and testing code for the FrozenLake environment.

## 📄 Report
- For the detailed experimental report, please refer to the PDF document in this directory.

## 📚 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. (Chapter 5: Monte Carlo Methods)

## 📅 Update Date
April 3, 2026