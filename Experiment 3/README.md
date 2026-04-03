# Experiment 3: Dynamic Programming and Markov Decision Process (MDP)

This experiment mainly focuses on model-based reinforcement learning methods by implementing classic dynamic programming core algorithms to solve for the optimal policy and optimal state-value function of a Markov Decision Process (MDP).

## 📝 Experiment Overview

1. **Environment Construction**: Familiarize with and use a custom grid world (FrozenLake environment), mastering the acquisition and definition of state transition probabilities and reward matrices.
2. **Algorithm Implementation**:
   - **Policy Evaluation**: Calculate the state-value function under a given deterministic policy by iterating the Bellman expectation equation.
   - **Policy Iteration**: Alternate between policy evaluation and policy improvement steps until the policy converges to the optimal one.
   - **Value Iteration**: Directly solve for the optimal state-value function by repeatedly applying the Bellman optimality operator, and then extract the optimal policy.
3. **Result Comparison and Analysis**: Compare the differences between policy iteration and value iteration in terms of convergence steps, computational overhead, and performance, and analyze the impact of different hyperparameters (such as the discount factor) on algorithm results and execution efficiency.

## 💻 Code
- Contains the Python implementation files for the core dynamic programming algorithms (Policy Evaluation, Policy Iteration, Value Iteration), as well as the specific reinforcement learning environment code used to test the algorithms.

## 📄 Report
- For the detailed experimental report, please refer to the PDF document in this directory.

## 📚 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. (Chapter 4: Dynamic Programming)

## 📅 Update Date
April 3, 2026