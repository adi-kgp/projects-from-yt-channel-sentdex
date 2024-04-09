# pip install gym[box2d]
import gymnasium as gym
from stable_baselines3 import PPO, A2C

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

models_dir = f'models/PPO-1712584694'
model_path = f"{models_dir}/290000.zip"

model = A2C.load(model_path, env=env)

# Enjoy trained agent
vec_env = model.get_env()
episodes = 10
for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        vec_env.render("human")
        action, _states = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        # print(reward)
    
env.close()
