"""
对比SARSA和Q-Learning算法在FrozenLake问题上的表现
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from SARSA_FrozenLake import SARSA_Agent
from QLearning_FrozenLake import QLearning_Agent

def compare_algorithms(num_episodes=10000, num_runs=5):
    """对比两种算法的性能"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    sarsa_rewards_all = []
    qlearning_rewards_all = []
    
    for run in range(num_runs):
        print(f"\n===== 运行 {run+1}/{num_runs} =====")
        
        # SARSA
        print("训练 SARSA...")
        sarsa_agent = SARSA_Agent(env, alpha=0.1, gamma=0.99, epsilon=0.5)
        sarsa_rewards, _, _ = sarsa_agent.train(num_episodes)
        sarsa_rewards_all.append(sarsa_rewards)
        
        # Q-Learning
        print("训练 Q-Learning...")
        qlearning_agent = QLearning_Agent(env, alpha=0.1, gamma=0.99, epsilon=0.5)
        qlearning_rewards, _, _ = qlearning_agent.train(num_episodes)
        qlearning_rewards_all.append(qlearning_rewards)
    
    # 计算平均性能
    sarsa_avg = np.mean(sarsa_rewards_all, axis=0)
    qlearning_avg = np.mean(qlearning_rewards_all, axis=0)
    
    # 绘制对比图
    plt.figure(figsize=(14, 5))
    
    # 平滑曲线
    window = 100
    sarsa_smooth = np.convolve(sarsa_avg, np.ones(window)/window, mode='valid')
    qlearning_smooth = np.convolve(qlearning_avg, np.ones(window)/window, mode='valid')
    
    plt.subplot(1, 2, 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(sarsa_smooth, label='SARSA', color='blue')
    plt.plot(qlearning_smooth, label='Q-Learning', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'SARSA vs Q-Learning 学习曲线 (平均{num_runs}次运行)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sarsa_success = np.cumsum(sarsa_avg) / (np.arange(len(sarsa_avg)) + 1)
    qlearning_success = np.cumsum(qlearning_avg) / (np.arange(len(qlearning_avg)) + 1)
    plt.plot(sarsa_success, label='SARSA', color='blue')
    plt.plot(qlearning_success, label='Q-Learning', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('累计成功率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Compare_SARSA_QLearning.png', dpi=150)
    plt.show()
    
    # 打印最终统计
    print("\n" + "=" * 60)
    print("最终性能对比 (最后1000回合平均)")
    print("=" * 60)
    print(f"SARSA      - 平均奖励: {np.mean(sarsa_avg[-1000:]):.4f}")
    print(f"Q-Learning - 平均奖励: {np.mean(qlearning_avg[-1000:]):.4f}")
    
    # 测试最终策略
    print("\n测试最终策略 (各100回合):")
    sarsa_agent = SARSA_Agent(env, alpha=0.1, gamma=0.99, epsilon=0.5)
    sarsa_agent.train(num_episodes)
    sarsa_success, _ = sarsa_agent.test(100)
    
    qlearning_agent = QLearning_Agent(env, alpha=0.1, gamma=0.99, epsilon=0.5)
    qlearning_agent.train(num_episodes)
    qlearning_success, _ = qlearning_agent.test(100)
    
    print(f"SARSA      - 成功率: {sarsa_success*100:.1f}%")
    print(f"Q-Learning - 成功率: {qlearning_success*100:.1f}%")
    
    env.close()
    return sarsa_avg, qlearning_avg


if __name__ == '__main__':
    print("=" * 60)
    print("SARSA vs Q-Learning 算法对比实验")
    print("环境: FrozenLake-v1 (滑动冰面)")
    print("=" * 60)
    
    compare_algorithms(num_episodes=10000, num_runs=3)
