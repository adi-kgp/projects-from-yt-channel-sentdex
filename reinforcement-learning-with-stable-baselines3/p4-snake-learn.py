# pip install gym[box2d]
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time
from snakeenv import SnakeEnv

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    
env = SnakeEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir) # multilayerperceptron
# model.learn(total_timesteps=10000, progress_bar=True)

TIMESTEPS = 10000
for i in range(1,10):
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close()
