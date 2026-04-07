"""
SARSA算法解决FrozenLake冰湖问题
参考: TD_SARSA_RL.py
"""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import copy

class SARSA_Agent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.5):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # 初始化Q值表
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        # 初始化策略
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        
    def reset(self):
        """重置Q值表"""
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.epsilon = 0.5
        
    def sample_action(self, state):
        """根据epsilon-greedy策略采样动作"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.qvalue[state, :])
    
    def update_policy(self):
        """更新epsilon-greedy策略"""
        for s in range(self.n_states):
            self.policy[s, :] = self.epsilon / self.n_actions
            best_action = np.argmax(self.qvalue[s, :])
            self.policy[s, best_action] += (1 - self.epsilon)
    
    def get_greedy_policy(self):
        """获取贪婪策略"""
        greedy_policy = np.zeros((self.n_states, self.n_actions))
        actions = ['←', '↓', '→', '↑']
        print("\n最优贪婪策略:")
        for s in range(self.n_states):
            best_action = np.argmax(self.qvalue[s, :])
            greedy_policy[s, best_action] = 1
            print(f"状态 {s:2d} -> {actions[best_action]}")
        return greedy_policy
    
    def train(self, num_episodes=10000):
        """SARSA训练"""
        rewards_history = []
        episode_lengths = []
        q_changes = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            action = self.sample_action(state)
            total_reward = 0
            steps = 0
            q_former = copy.deepcopy(self.qvalue)
            
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # SARSA: 使用实际采取的下一个动作来更新
                next_action = self.sample_action(next_state)
                
                if done:
                    td_target = reward
                else:
                    td_target = reward + self.gamma * self.qvalue[next_state, next_action]
                
                # Q值更新
                td_error = td_target - self.qvalue[state, action]
                self.qvalue[state, action] += self.alpha * td_error
                
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
            
            # 更新策略和探索率
            self.update_policy()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            rewards_history.append(total_reward)
            episode_lengths.append(steps)
            
            # 记录Q值变化
            if episode % 100 == 0:
                q_change = np.linalg.norm(self.qvalue - q_former)
                q_changes.append(q_change)
                if episode % 1000 == 0:
                    avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                    print(f"Episode {episode}, 平均奖励: {avg_reward:.3f}, Epsilon: {self.epsilon:.4f}")
        
        return rewards_history, episode_lengths, q_changes
    
    def test(self, num_episodes=100, render=False):
        """测试训练好的策略"""
        success_count = 0
        total_rewards = []
        
        for _ in range(num_episodes):
            if render:
                env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
                state, _ = env.reset()
            else:
                state, _ = self.env.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action = np.argmax(self.qvalue[state, :])
                if render:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            
            total_rewards.append(total_reward)
            if total_reward > 0:
                success_count += 1
            
            if render:
                env.close()
        
        return success_count / num_episodes, np.mean(total_rewards)


def visualize_policy(qvalue, env_size=4):
    """可视化策略"""
    actions = ['←', '↓', '→', '↑']
    policy_grid = []
    
    print("\n策略可视化 (4x4网格):")
    print("-" * 25)
    for i in range(env_size):
        row = []
        for j in range(env_size):
            state = i * env_size + j
            best_action = np.argmax(qvalue[state, :])
            row.append(actions[best_action])
        policy_grid.append(row)
        print(f"| {' | '.join(row)} |")
        print("-" * 25)
    
    return policy_grid


def plot_results(rewards, title="SARSA Learning Curve"):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, color='blue')
    # 滑动平均
    window = 100
    if len(rewards) >= window:
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), avg_rewards, color='red', label=f'{window}回合平均')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{title} - 每回合奖励')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cumulative_success = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.plot(cumulative_success, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'{title} - 累计成功率')
    
    plt.tight_layout()
    plt.savefig('SARSA_FrozenLake_results.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    # 创建环境 (is_slippery=True表示冰面滑动)
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # 创建SARSA智能体
    agent = SARSA_Agent(env, alpha=0.1, gamma=0.99, epsilon=0.5)
    
    print("=" * 50)
    print("SARSA算法解决FrozenLake冰湖问题")
    print("=" * 50)
    
    # 训练
    print("\n开始训练...")
    rewards, lengths, q_changes = agent.train(num_episodes=10000)
    
    # 显示结果
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    
    # 获取并显示最优策略
    greedy_policy = agent.get_greedy_policy()
    
    # 可视化策略网格
    visualize_policy(agent.qvalue)
    
    # 显示Q值表
    print("\nQ值表:")
    print(np.around(agent.qvalue, 3))
    
    # 测试策略
    print("\n测试策略 (100回合)...")
    success_rate, avg_reward = agent.test(num_episodes=100)
    print(f"成功率: {success_rate*100:.1f}%")
    print(f"平均奖励: {avg_reward:.3f}")
    
    # 绘制学习曲线
    plot_results(rewards, "SARSA")
    
    env.close()
