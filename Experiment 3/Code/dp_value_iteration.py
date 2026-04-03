import time
import gymnasium as gym
import numpy as np

class DP_Value_Iter:
    def __init__(self):
        self.actions = [0, 1]
        # 状态离散化
        self.n_bins = [3, 3, 3, 3]
        self.state_bins = self._create_bins()
        self.states = list(range(81))
        self.v = [0.0] * len(self.states)
        self.pi = {state: 0 for state in self.states}
        self.gamma = 1.0

    def _create_bins(self):
        bins = [
            np.linspace(-4.8, 4.8, self.n_bins[0] + 1),
            np.linspace(-3.0, 3.0, self.n_bins[1] + 1),
            np.linspace(-0.418, 0.418, self.n_bins[2] + 1),
            np.linspace(-2.0, 2.0, self.n_bins[3] + 1)
        ]
        return bins

    def _continuous_to_discrete(self, state):
        discrete = 0
        for i in range(4):
            discrete = discrete * self.n_bins[i] + (np.digitize(state[i], self.state_bins[i]) - 1)
        return discrete

    # ✅ 纯离散转移，无环境交互
    def transform(self, state, action):
        next_state = (state + action) % len(self.states)
        reward = 1.0 if next_state != 0 else 0.0
        done = next_state == 0
        return next_state, reward, done

    # 值迭代（二合一评估+改进）
    def value_iteration(self):
        print("开始值迭代...")
        for _ in range(1000):
            delta = 0.0
            for state in self.states:
                max_val = -float('inf')
                best_action = 0
                # 遍历动作，取最大值
                for action in self.actions:
                    next_s, r, _ = self.transform(state, action)
                    val = r + self.gamma * self.v[next_s]
                    if val > max_val:
                        max_val = val
                        best_action = action
                delta += abs(max_val - self.v[state])
                self.v[state] = max_val
                self.pi[state] = best_action
            if delta < 1e-6:
                print("值迭代收敛！")
                break

if __name__ == "__main__":
    # 初始化DP算法
    dp = DP_Value_Iter()
    dp.value_iteration()

    # ✅ 测试阶段：环境交互（先reset）
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    total_reward = 0
    step = 0

    print("\n===== 值迭代策略测试 =====")
    while True:
        discrete_state = dp._continuous_to_discrete(state)
        action = dp.pi[discrete_state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        state = next_state

        print(f"步数：{step} | 累计奖励：{total_reward}")
        if terminated or truncated or step >= 500:
            print(f"测试结束！坚持步数：{step}")
            break
        time.sleep(0.05)
    env.close()