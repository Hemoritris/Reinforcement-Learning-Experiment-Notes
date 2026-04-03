import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="human")

state, info = env.reset()
total_steps = 0

while True:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    total_steps += 1
    
    if terminated or truncated:
        print(f"游戏结束！总步数：{total_steps}")
        break
    
    state = next_state

env.close()