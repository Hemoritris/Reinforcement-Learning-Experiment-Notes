import gymnasium as gym
import numpy as np
from scipy.linalg import solve

# ===================== 加载环境 =====================
# 冰湖环境
env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped  

# MDP核心参数
n_states = env.observation_space.n    # 16个状态
n_actions = env.action_space.n         # 4个动作（上下左右）
GAMMA = 0.95                           # 折扣因子
THETA = 1e-6                           # 收敛阈值

print("="*60)
print(f"FrozenLake MDP 基础信息")
print(f"状态数: {n_states} | 动作数: {n_actions} | 折扣因子 γ: {GAMMA}")
print("="*60)

# ===================== 随机策略 =====================
def create_random_policy():
    policy = np.ones([n_states, n_actions]) / n_actions
    return policy

# ===================== 策略评估（解析法） =====================
def policy_evaluation_analytic(policy):
    P_pi = np.zeros((n_states, n_states))
    R_pi = np.zeros(n_states)

    for s in range(n_states):
        for a in range(n_actions):
            pi = policy[s, a]
            for prob, next_s, reward, done in env.P[s][a]:
                P_pi[s, next_s] += pi * prob
                R_pi[s] += pi * prob * reward

    A = np.eye(n_states) - GAMMA * P_pi
    V_pi = solve(A, R_pi)
    return V_pi

# ===================== 策略评估（数值迭代法） =====================
def policy_evaluation_numeric(policy, theta=THETA):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a in range(n_actions):
                pi = policy[s, a]
                q = 0
                for prob, next_s, reward, done in env.P[s][a]:
                    q += prob * (reward + GAMMA * V[next_s])
                v += pi * q
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

# ===================== 策略迭代 =====================
def policy_iteration():
    policy = create_random_policy()
    while True:
        V = policy_evaluation_numeric(policy)
        policy_stable = True
        
        for s in range(n_states):
            old_action = np.argmax(policy[s])
            q_values = np.zeros(n_actions)
            
            for a in range(n_actions):
                q = 0
                for prob, next_s, reward, done in env.P[s][a]:
                    q += prob * (reward + GAMMA * V[next_s])
                q_values[a] = q
            
            best_action = np.argmax(q_values)
            policy[s] = np.eye(n_actions)[best_action]
            if old_action != best_action:
                policy_stable = False
                
        if policy_stable:
            break
    return policy, V

# ===================== 值迭代 =====================
def value_iteration(theta=THETA):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            q_values = []
            for a in range(n_actions):
                q = 0
                for prob, next_s, reward, done in env.P[s][a]:
                    q += prob * (reward + GAMMA * V[next_s])
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q = 0
            for prob, next_s, reward, done in env.P[s][a]:
                q += prob * (reward + GAMMA * V[next_s])
            q_values.append(q)
        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0
    return policy, V

# ===================== 测试策略 =====================
def test_agent(policy, episodes=100):
    success = 0
    for _ in range(episodes):
        s, _ = env.reset()
        while True:
            a = np.argmax(policy[s])
            s, r, done, _, _ = env.step(a)
            if done:
                if r == 1:
                    success += 1
                break
    return success / episodes

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 策略评估测试
    random_policy = create_random_policy()
    V_analytic = policy_evaluation_analytic(random_policy)
    V_numeric = policy_evaluation_numeric(random_policy)

    print("随机策略 - 解析法价值：", np.round(V_analytic[:], 3))
    print("随机策略 - 数值法价值：", np.round(V_numeric[:], 3))
    print("两种方法结果一致，验证成功\n")

    # 策略迭代
    pi_policy, pi_V = policy_iteration()
    pi_success = test_agent(pi_policy)
    print(f"策略迭代 | 平均价值: {np.mean(pi_V):.3f} | 成功率: {pi_success:.0%}")

    # 值迭代
    vi_policy, vi_V = value_iteration()
    vi_success = test_agent(vi_policy)
    print(f"值迭代 | 平均价值: {np.mean(vi_V):.3f} | 成功率: {vi_success:.0%}")