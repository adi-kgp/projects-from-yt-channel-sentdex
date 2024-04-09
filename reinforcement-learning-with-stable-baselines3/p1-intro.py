# pip install gym[box2d]
import gymnasium as gym
from stable_baselines3 import PPO

# create environment
env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1) # multilayerperceptron policy
# Train the agent and display a progress bar
model.learn(total_timesteps=10000, progress_bar=True)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        # print(reward)
    
env.close()
