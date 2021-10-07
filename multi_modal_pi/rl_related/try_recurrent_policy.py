import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from gym.envs.registration import register
import os
import numpy as np
import logging
import random




from SensorFusion.rl_related.helpers_RL import *



register(
  id='MultiGoalSnakePath-v0',
  entry_point='environment:GoalBasedWorld')

register(
  id='MultiGoalSnakePath-v1',
  entry_point='environment:GoalBasedPreprocessedWorld')

# Parallel environments
env = make_vec_env('CartPole-v1', n_envs=4)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")
