# Reinforcement-Learning-Experiment-Notes

本项目记录了我在学习强化学习（Reinforcement Learning）过程中的实验内容。仓库包含了各个强化学习算法的Python实现代码、详细的实验报告以及相关的参考资料。

## 📖 仓库简介

在学习强化学习的过程中，我通过编写代码并在经典环境（如 FrozenLake、自定义环境等）中进行测试，加深了对各种算法的理解。本仓库对这些实验进行了归档，每一个实验通常包含以下内容：
- **算法代码**：核心算法的 Python 实现。
- **环境代码**：测试算法使用的具体环境。
- **实验报告**：对实验过程、理论推导及结果分析的总结。
- **参考资料**：学习过程中参考的文献（如 Sutton & Barto 的经典教材《Reinforcement Learning: An Introduction》等）、代码等。

## 🗂️ 目录结构

目前仓库包含以下核心模块（会随着学习进度不断更新）：

### 0. 基础强化学习与环境建模
- **Experiment 1**：4x4 井字棋强化学习实验，围绕自定义对弈环境、智能体自博弈训练与对抗评估展开。
- **Experiment 2**：CartPole-v1 的 MDP 建模分析，完成状态、动作、转移、奖励与折扣因子的系统化描述。

### 1. 动态规划（Dynamic Programming）
在这部分实验中，实现了基于模型的经典动态规划算法：
- **策略迭代（Policy Iteration）**
- **价值迭代（Value Iteration）**

### 2. 蒙特卡洛方法（Monte Carlo Methods）
在不需要完整环境动态模型的情况下，通过采样进行强化学习：
- **On-policy MC 控制**
- **Off-policy MC 控制**
- 对应的强化学习抽象和基类实现。

### 3. 时序差分算法（Temporal Difference Learning）
- **SARSA 算法**：基于当前策略的同策略（On-policy）TD 控制。
- **Q-Learning 算法**：基于最优策略的异策略（Off-policy）TD 控制。
- **算法对比分析**：对两种经典 TD 算法在收敛速度、当前表现及最佳表现等方面进行可视化对比评估。

### 4. 深度强化学习（Deep Reinforcement Learning）
- **Deep Q-Network（DQN）**：在 Atari `ALE/Breakout-v5` 环境中实现基于卷积神经网络的值函数逼近。
- **经验回放（Experience Replay）**：通过回放缓存打破样本相关性，提高训练稳定性。
- **目标网络（Target Network）**：周期性同步目标 Q 网络，减弱训练震荡。
- **训练可视化与策略演示**：提供训练曲线、模型检查点以及策略运行视频，便于分析学习过程与最终表现。


## 📖 实验报告
包含了针对每次实验内容撰写的详细报告，详见各章节的 PDF 文件。

## 🚀 说明

本仓库内容仅为学习强化学习算法过程中的一些实践，可能包含一些错误或者不足之处，如果要做参考请自行检查。

