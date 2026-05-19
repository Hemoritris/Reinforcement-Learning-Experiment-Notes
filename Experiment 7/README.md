# Experiment 7: Policy Gradient on CartPole

This experiment implements a policy-gradient agent with the REINFORCE algorithm to solve the classic Gym/Gymnasium `CartPole-v1` control task. The project covers stochastic policy modeling, trajectory sampling, discounted-return estimation, batched policy updates, model saving, evaluation, and training-curve visualization.

## 📘 Experiment Overview

1. **Environment Construction**: Use the `CartPole-v1` environment, where the agent controls a cart by choosing left/right actions to keep a pole balanced for as long as possible.
2. **Algorithm Implementation**:
   - **Policy Network**: Use a multilayer perceptron to map the 4-dimensional state vector to action logits.
   - **REINFORCE Update**: Optimize the stochastic policy directly using sampled trajectories and discounted returns.
   - **Return Normalization**: Normalize returns to reduce the variance of policy-gradient estimates.
   - **Batched Training**: Collect multiple complete episodes before each policy update for more stable learning.
3. **Training and Evaluation**: Save the trained model, output a reward curve, and evaluate the learned policy with greedy action selection.

## 📂 Code Files

| File Name | Description |
|--------|------|
| `Code/policy_gradient_train.py` | REINFORCE training and evaluation script for `CartPole-v1`, including Gym/Gymnasium compatibility, model saving, and training-curve generation |
| `Code/policy_gradient_cartpole.pth` | Saved policy-network checkpoint after training |
| `Report/Experiment_7_Report.pdf` | Detailed experiment report |
| `Report/policy_gradient_training_curve.png` | Training curve showing episode reward and moving-average reward |

## 📄 Experiment Report

Please refer to `Report/Experiment_7_Report.pdf` in this directory for the detailed experiment report and result analysis.

## 🧠 Core Algorithm Formulas

- **Discounted Return**:
$$G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}$$

- **Policy Objective**:
$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]$$

- **REINFORCE Gradient**:
$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)G_t\right]$$

- **Training Loss**:
$$L(\theta)=-\frac{1}{N}\sum_{t=1}^{N}\log \pi_\theta(a_t|s_t)\hat{G}_t$$

## ⚙️ Experiment Parameters

| Parameter | Value |
|------|-----|
| Environment | `CartPole-v1` |
| Algorithm | REINFORCE |
| Episodes | 1,000 |
| Batch Size | 10 episodes |
| Learning Rate | 1e-3 |
| Discount Factor $\gamma$ | 0.99 |
| Hidden Dimension | 128 |
| Optimizer | Adam |
| Solved Window | 50 episodes |
| Solved Score | 475.0 |
| Evaluation Episodes | 5 |
| Evaluation Average Reward | 500.00 |
| Device | CPU |

## ▶️ Usage

Train the policy:

```powershell
cd "Experiment 7\Code"
python .\policy_gradient_train.py --cpu
```

Evaluate the saved policy:

```powershell
python .\policy_gradient_train.py --eval --eval-episodes 5 --model-path .\policy_gradient_cartpole.pth --cpu
```

Render the learned policy:

```powershell
python .\policy_gradient_train.py --eval --render --model-path .\policy_gradient_cartpole.pth --cpu
```

## 📚 References

- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Gymnasium Documentation: Classic Control Environments.
- PyTorch Documentation: `torch.distributions.Categorical`.

## 🗓️ Update Date

May 19, 2026
