from gym.envs.registration import register
import gym
import os
import numpy as np
import logging
# from stable_baselines3.common.torch_layers import FeedForwardPolicy, register_policy
from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.her import HerReplayBuffer
import torch as tch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.monitor import Monitor
from helpers_RL import *

register(
  id='MultiGoalSnakePath-v0',
  entry_point='environment:GoalBasedWorld')

register(
  id='MultiGoalSnakePath-v1',
  entry_point='environment:GoalBasedPreprocessedWorld')

SEED=0
EVAL_EVERY=5000
TENSORBOARD_LOG_FOLDER = "./logs/seed{}/tensorboard/".format(SEED)
os.makedirs(TENSORBOARD_LOG_FOLDER, exist_ok=True)


goal_selection_strategy = 'future'

env_name = "MultiGoalSnakePath-v1"
env = gym.make(env_name, seed=SEED)
env = Monitor(env, TENSORBOARD_LOG_FOLDER)

eval_env = gym.make(env_name, seed=SEED)
eval_env = Monitor(eval_env, TENSORBOARD_LOG_FOLDER)

# fig_path = './logs/seed{}/figures/'.format(SEED)
# os.makedirs(fig_path, exist_ok=True)
fig_name_prefix = 'trajectory_figures_'

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=EVAL_EVERY, save_path='./logs/seed{}/saves/'.format(SEED),)
policy_eval_callback = PolicyEvaluationCallback(eval_env, save_freq=EVAL_EVERY, fig_name_prefix=fig_name_prefix,)

model = TD3(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    buffer_size=10000*20,

    learning_rate=3e-4,
    gamma=0.95,
    batch_size=1024,

    policy_kwargs={
        'net_arch': [512, 256, 128, 32],
        # 'features_dim': 256,
        "features_extractor_class": CustomCombinedExtractor,
        },
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        online_sampling=True,
        max_episode_length=30,
    ),
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG_FOLDER,
)

# Works fine, so stick with it
# model.learn(total_timesteps=100)
# model.learn(total_timesteps=1000000)
model.learn(total_timesteps=400000,
        callback=CallbackList([checkpoint_callback, policy_eval_callback]),
        tb_log_name='default',
        )
model.save('./logs/{}/final_model'.format(SEED))

os.makedirs('out/tests/multi_goal_baseline/{}'.format(env_name), exist_ok=True)
logging.critical('start evaluating')
for traj in range(10):
    obs = env.reset()
    start_room, start_pos = env.agent_room, env.agent_position
    reward_room, reward_pos = env.reward_room, env.reward_position
    logging.critical('[traj{}] initial position : room {}, xy {}; target at room {}, xy {}'.format(traj, env.agent_room, env.agent_position, env.reward_room, env.reward_position))
    actions = np.zeros((20, 2))
    for t in range(20):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        actions[t] = info['rectified_action']#.detach().cpu().numpy()
        logging.critical('[traj{}] action: {}; rectified action {}, reward {}, done {}'.format(traj, action, info['rectified_action'], reward, done))
        logging.critical('[traj{}] current position : room {}, xy {}'.format(traj, env.agent_room, env.agent_position))
        if done:
            break
    # logging.critical('exited inner loop')
    env.plot_trajectory(actions, start_room=start_room, start_pos=start_pos, reward_room=reward_room, reward_pos=reward_pos, save_file='out/tests/multi_goal_baseline/{}/traj_{}.pdf'.format(env_name, traj))
