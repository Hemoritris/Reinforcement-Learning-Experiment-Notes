import numpy as np
import copy
import gymnasium as gym

class FrozenLakeMC:
    def __init__(self, is_slippery=False):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n  # 16个状态
        self.n_actions = self.env.action_space.n      # 4个动作
        
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.C = np.zeros((self.n_states, self.n_actions))
        
        self.states = np.arange(self.n_states)
        self.actions = np.arange(self.n_actions)
        self.gamma = 0.99
        
        self.behaviour_Pi = (1.0 / self.n_actions) * np.ones((self.n_states, self.n_actions))
        self.target_Pi = np.zeros((self.n_states, self.n_actions))
        for i in range(self.n_states):
            j = np.random.choice(self.actions)
            self.target_Pi[i, j] = 1
        
        self.Greedy_Pi = np.zeros((self.n_states, self.n_actions))
        self.cur_state = 0
        self.epsilon = 0.5

    def reset(self):
        self.qvalue = np.zeros((self.n_states, self.n_actions))
        self.C = np.zeros((self.n_states, self.n_actions))

    def sample_action(self, state):
        action = np.random.choice(self.actions, p=self.behaviour_Pi[state, :])
        return action

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def update_target_policy(self):
        epsilon = max(self.epsilon / 10, 0.01)
        for i in range(self.n_states):
            self.target_Pi[i, :] = epsilon / self.n_actions
            max_num = np.argmax(self.qvalue[i, :])
            self.target_Pi[i, max_num] = epsilon / self.n_actions + (1 - epsilon)

    def update_behaviour_policy(self):
        for i in range(self.n_states):
            self.behaviour_Pi[i, :] = self.epsilon / self.n_actions
            max_num = np.argmax(self.qvalue[i, :])
            self.behaviour_Pi[i, max_num] = self.epsilon / self.n_actions + (1 - self.epsilon)

    def get_greedy_policy(self):
        for i in range(self.n_states):
            self.Greedy_Pi[i, :] = 0
            max_num = np.argmax(self.qvalue[i, :])
            self.Greedy_Pi[i, max_num] = 1
        return self.Greedy_Pi

    def Off_MC_learning(self, num_episodes=10000):
        self.update_target_policy()
        self.update_behaviour_policy()
        
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
            W = 1.0
            for i in reversed(range(len(state_traj))):
                g = self.gamma * g + reward_traj[i]
                
                self.C[state_traj[i], action_traj[i]] += W
                
                self.qvalue[state_traj[i], action_traj[i]] += \
                    (W / self.C[state_traj[i], action_traj[i]]) * \
                    (g - self.qvalue[state_traj[i], action_traj[i]])
                
                if self.behaviour_Pi[state_traj[i], action_traj[i]] == 0:
                    break
                W = W * self.target_Pi[state_traj[i], action_traj[i]] / \
                    self.behaviour_Pi[state_traj[i], action_traj[i]]
                
                if W == 0:
                    break

            if (num + 1) % 1000 == 0:
                self.epsilon = max(self.epsilon * 0.95, 0.1)
                self.update_target_policy()
                self.update_behaviour_policy()
                print(f"Episode {num + 1}, epsilon: {self.epsilon:.4f}")

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
    
    print("开始离策略蒙特卡洛学习...")
    agent.Off_MC_learning(num_episodes=10000)
    
    agent.get_greedy_policy()
    print("\n贪婪策略:\n", agent.Greedy_Pi)
    print("\n估计Q值函数:\n", np.around(agent.qvalue, 3))
    
    agent.print_policy()
    
    print("\n测试策略...")
    success_rate = agent.test_policy(num_tests=100)
    print(f"成功率: {success_rate * 100:.1f}%")
    
    print("\n累计权重:\n", np.around(agent.C, 1))
