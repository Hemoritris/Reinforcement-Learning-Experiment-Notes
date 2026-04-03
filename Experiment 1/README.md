# Experiment 1: 4x4 Tic-Tac-Toe Reinforcement Learning Experiment

In this experiment, the traditional 3x3 Tic-Tac-Toe game is extended and modified into a 4x4 Tic-Tac-Toe environment. The agents are trained for different numbers of episodes using reinforcement learning algorithms, and a detailed adversarial evaluation and analysis of the training results is conducted.

## 📝 Experiment Overview

1. **Environment Modification**: Reviewed and modified the reference 3x3 Tic-Tac-Toe code to successfully build a 4x4 Tic-Tac-Toe interactive environment.
2. **Model Training**: Trained the agents through self-play for $10^4$ (1e4), $10^5$ (1e5), and $10^6$ (1e6) episodes, respectively.
3. **Adversarial Evaluation**:
   - **Intra-group Evaluation**: Matches between the first-mover agent and second-mover agent from the same training amount.
   - **Inter-group Cross Evaluation**: Cross matches between agents with different training amounts (e.g., 1e4 vs 1e6).
4. **Result Analysis**: Compared, analyzed, and summarized the agents' win rates, strategic performance, and first-mover advantages under different volumes of training data.

## 💻 Code
- Contains the core Python code for the modified 4x4 Tic-Tac-Toe environment implementation, along with the reinforcement learning training and match evaluation.

## 📄 Report
- Please refer to the report document in this directory for detailed experimental notes, which include win-rate statistical charts, experimental derivation processes, and the final result analysis and summary.

## 📚 References
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. (The classic reinforcement learning example on Tic-Tac-Toe in Chapter 1)

## 📅 Update Date
April 3, 2026