import numpy as np
import copy
import gymnasium as gym

class FrozenLakeMC:
    def __init__(self, is_slippery=False):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n  # 16个状态
        self.n_actions = self.env.action_space.n      # 4个动作
        
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))
        
        self.states = np.arange(self.n_states)
        self.actions = np.arange(self.n_actions)
        self.gamma = 0.99
        
        self.Pi = (1.0 / self.n_actions) * np.ones((self.n_states, self.n_actions))
        self.Greedy_Pi = np.zeros((self.n_states, self.n_actions))
        self.cur_state = 0
        self.old_policy = np.ones((self.n_states, self.n_actions))
        self.epsilon = 0.5

    def reset(self):
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))

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
            max_num = np.argmax(self.qvalue[i, :])
            self.Pi[i, max_num] = 1

    def get_greedy_policy(self):
        for i in range(self.n_states):
            self.Greedy_Pi[i, :] = 0
            max_num = np.argmax(self.qvalue[i, :])
            self.Greedy_Pi[i, max_num] = 1
        return self.Greedy_Pi

    def update_epsilon_greedy(self):
        for i in range(self.n_states):
            self.Pi[i, :] = self.epsilon / self.n_actions
            max_num = np.argmax(self.qvalue[i, :])
            self.Pi[i, max_num] = self.epsilon / self.n_actions + (1 - self.epsilon)

    def MC_learning(self, num_episodes=10000):
        for num in range(num_episodes):
            state_traj = []
            action_traj = []
            reward_traj = []
            
            self.cur_state, _ = self.env.reset()
            done = False
            episode_num = 0
            max_steps = 100
            
            while not done and episode_num < max_steps:
                cur_action = self.sample_action(self.cur_state)
                state_traj.append(self.cur_state)
                action_traj.append(cur_action)
                
                next_state, reward, done = self.step(cur_action)
                reward_traj.append(reward)
                self.cur_state = next_state
                episode_num += 1

            g = 0
            visited = set()
            for i in reversed(range(len(state_traj))):
                g = self.gamma * g + reward_traj[i]
                state_action = (state_traj[i], action_traj[i])
                
                if state_action not in visited:
                    visited.add(state_action)
                    self.n[state_traj[i], action_traj[i]] += 1.0
                    self.qvalue[state_traj[i], action_traj[i]] = \
                        (self.qvalue[state_traj[i], action_traj[i]] * 
                         (self.n[state_traj[i], action_traj[i]] - 1) + g) / \
                        self.n[state_traj[i], action_traj[i]]

            if (num + 1) % 2000 == 0:
                self.old_policy = copy.deepcopy(self.Pi)
                self.update_epsilon_greedy()
                self.epsilon = max(self.epsilon * 0.9, 0.05)
                delta = np.linalg.norm(self.old_policy - self.Pi)
                print(f"Episode {num + 1}, Policy delta: {delta:.4f}, epsilon: {self.epsilon:.4f}")

    def test_policy(self, num_tests=100):
        test_env = gym.make('FrozenLake-v1', is_slippery=False)
        successes = 0
        for _ in range(num_tests):
            state, _ = test_env.reset()
            done = False
            steps = 0
            while not done and steps < 100:
                action = np.argmax(self.Greedy_Pi[state, :])
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
                best_action = np.argmax(self.Greedy_Pi[state, :])
                if state in [5, 7, 11, 12]:  # 冰洞
                    row += " H "
                elif state == 15:  # 目标
                    row += " G "
                elif state == 0:  # 起点
                    row += f"S{action_symbols[best_action]}"
                else:
                    row += f" {action_symbols[best_action]} "
            print(row)


if __name__ == '__main__':
    agent = FrozenLakeMC(is_slippery=False)
    agent.reset()
    
    print("初始策略: 均匀随机")
    print("开始同策略蒙特卡洛学习...")
    
    agent.MC_learning(num_episodes=10000)
    
    print("\n最终ε-贪婪策略:\n", np.around(agent.Pi, 2))
    print("\n估计Q值函数:\n", np.around(agent.qvalue, 3))
    
    agent.get_greedy_policy()
    agent.print_policy()
    
    print("\n测试策略...")
    success_rate = agent.test_policy(num_tests=100)
    print(f"成功率: {success_rate * 100:.1f}%")
    
    print("\n访问频次:\n", agent.n)
    print("\n探索因子:", agent.epsilon)
    print("\n贪婪策略:\n", agent.get_greedy_policy())
