import numpy as np
import copy
import gymnasium as gym

class FrozenLakeMC:
    def __init__(self, env_name='FrozenLake-v1', is_slippery=False):
        self.env = gym.make(env_name, is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n  # 16个状态
        self.n_actions = self.env.action_space.n      # 4个动作: 0左 1下 2右 3上
        
        # 初始化行为值函数
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))
        
        self.states = np.arange(self.n_states)
        self.actions = np.arange(self.n_actions)
        self.gamma = 0.99
        
        # 初始化策略
        self.Pi = (1.0 / self.n_actions) * np.ones((self.n_states, self.n_actions))
        self.old_policy = np.ones((self.n_states, self.n_actions))
        self.cur_state = 0

    def reset(self):
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))

    def explore_init(self):
        self.cur_state, _ = self.env.reset()
        a0 = np.random.choice(self.actions)
        return self.cur_state, a0

    def sample_action(self, state):
        action = np.random.choice(self.actions, p=self.Pi[state, :])
        return action

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def update_policy(self):
        for i in range(self.n_states):
            self.Pi[i, :] = 0
            max_action = np.argmax(self.qvalue[i, :])
            self.Pi[i, max_action] = 1

    def update_epsilon_greedy(self, epsilon=0.1):
        for i in range(self.n_states):
            self.Pi[i, :] = epsilon / self.n_actions
            max_action = np.argmax(self.qvalue[i, :])
            self.Pi[i, max_action] = 1 - epsilon + epsilon / self.n_actions

    def MC_learning(self, num_episodes=10000, policy_update_interval=500):
        for episode in range(num_episodes):
            # 采样一条轨迹
            state_traj = []
            action_traj = []
            reward_traj = []
            
            self.cur_state, _ = self.env.reset()
            
            cur_action = np.random.choice(self.actions)
            
            done = False
            step_count = 0
            max_steps = 100
            
            while not done and step_count < max_steps:
                state_traj.append(self.cur_state)
                action_traj.append(cur_action)
                
                next_state, reward, done = self.step(cur_action)
                reward_traj.append(reward)
                
                self.cur_state = next_state
                cur_action = self.sample_action(self.cur_state)
                step_count += 1

            g = 0
            visited = set()
            for i in reversed(range(len(state_traj))):
                g = self.gamma * g + reward_traj[i]
                state_action = (state_traj[i], action_traj[i])
                
                if state_action not in visited:
                    visited.add(state_action)
                    self.n[state_traj[i], action_traj[i]] += 1.0
                    alpha = 1.0 / self.n[state_traj[i], action_traj[i]]
                    self.qvalue[state_traj[i], action_traj[i]] += \
                        alpha * (g - self.qvalue[state_traj[i], action_traj[i]])


            if (episode + 1) % policy_update_interval == 0:
                self.old_policy = copy.deepcopy(self.Pi)
                self.update_epsilon_greedy(epsilon=0.1)
                delta = np.linalg.norm(self.old_policy - self.Pi)
                print(f"Episode {episode + 1}, Policy change: {delta:.4f}")

    def test_policy(self, num_tests=100, render=False):
        if render:
            test_env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
        else:
            test_env = gym.make('FrozenLake-v1', is_slippery=False)
        
        successes = 0
        for _ in range(num_tests):
            state, _ = test_env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                action = np.argmax(self.Pi[state, :])
                state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                steps += 1
                if reward == 1.0:
                    successes += 1
        
        test_env.close()
        return successes / num_tests

    def print_policy(self):
        action_symbols = ['←', '↓', '→', '↑']
        print("\n学习到的策略:")
        for i in range(4):
            row = ""
            for j in range(4):
                state = i * 4 + j
                best_action = np.argmax(self.Pi[state, :])
                # 冰洞和目标位置特殊标记
                if state in [5, 7, 11, 12]:  # 冰洞
                    row += " H "
                elif state == 15:  # 目标
                    row += " G "
                else:
                    row += f" {action_symbols[best_action]} "
            print(row)


if __name__ == '__main__':
    agent = FrozenLakeMC(is_slippery=False)
    
    print("初始策略: 均匀随机")
    print("开始蒙特卡洛学习...")
    
    agent.MC_learning(num_episodes=10000, policy_update_interval=1000)
    
    print("\n最终Q值函数:")
    print(agent.qvalue)
    
    agent.print_policy()
    
    print("\n测试策略...")
    success_rate = agent.test_policy(num_tests=100)
    print(f"成功率: {success_rate * 100:.1f}%")
    
    print("\n访问频次:")
    print(agent.n)
