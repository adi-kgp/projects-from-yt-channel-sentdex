from snakeenv import SnakeEnv
from stable_baselines3 import PPO
import gym

models_dir = "models/PPO-1712644514"

env = SnakeEnv()
env.reset()

model_path = f"{models_dir}/290000.zip"

model = PPO.load(model_path, env=env)


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
        #print(reward)
    
env.close()