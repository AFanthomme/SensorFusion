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
from matplotlib.colors import hsv_to_rgb

from SensorFusion.rl_related.helpers_RL import *

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

plot_env = gym.make(env_name, seed=SEED)
plot_env = Monitor(plot_env, TENSORBOARD_LOG_FOLDER)

# fig_path = './logs/seed{}/figures/'.format(SEED)
# os.makedirs(fig_path, exist_ok=True)
fig_name_prefix = 'trajectory_figures_'

# Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(save_freq=EVAL_EVERY, save_path='./logs/seed{}/saves/'.format(SEED),)
# policy_eval_callback = PolicyEvaluationCallback(eval_env, save_freq=EVAL_EVERY, fig_name_prefix=fig_name_prefix,)

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

model = TD3.load('./logs/{}/final_model'.format(SEED), env)
# model = TD3.load('logs/seed0/saves/rl_model_10000_steps', env)
# model = TD3.load('logs/seed0/saves/rl_model_400000_steps', env)

os.makedirs('out/tests/policy/{}'.format(env_name), exist_ok=True)
logging.critical('start evaluating')
# frame_idx = 0

reward_resolution = 10

reward_rooms = [0] * reward_resolution + [1] * reward_resolution + [2] * reward_resolution + [3] * reward_resolution + [4] * reward_resolution + [5] * reward_resolution
reward_positions = np.zeros((reward_resolution * 6, 2))

f  = .6
reward_positions[:reward_resolution, 0] = np.linspace(-f * env.scale, env.scale, reward_resolution)
reward_positions[reward_resolution:2*reward_resolution, 0] = np.linspace(-env.scale, env.scale, reward_resolution)
reward_positions[2*reward_resolution:int(2.5*reward_resolution), 0] = np.linspace(-f * env.scale, f*env.scale, reward_resolution//2)
reward_positions[int(2.5*reward_resolution):3*reward_resolution, 0] = f*env.scale
reward_positions[int(2.5*reward_resolution):3*reward_resolution, 1] = np.linspace(0, env.scale, reward_resolution//2)
reward_positions[3*reward_resolution:int(3.5*reward_resolution), 0] = f*env.scale
reward_positions[3*reward_resolution:int(3.5*reward_resolution), 1] = np.linspace(-f * env.scale, 0, reward_resolution//2)
reward_positions[int(3.5*reward_resolution):4*reward_resolution, 0] = np.linspace(f * env.scale, -env.scale, reward_resolution//2)
reward_positions[4*reward_resolution:5*reward_resolution, 0] = np.linspace(env.scale, -env.scale, reward_resolution)
reward_positions[5*reward_resolution:6*reward_resolution, 0] = np.linspace(env.scale, -f * env.scale, reward_resolution)


frame_idx = 0
for reward_room, reward_pos in zip(reward_rooms, reward_positions):
    logging.critical(frame_idx)
    resolution = 6
    _ = env.reset()
    # env.reward_room, env.reward_position = deepcopy(reward_room), deepcopy(reward_pos)
    env.set_reward_position(reward_room, deepcopy(reward_pos))
    plot_env.set_reward_position(reward_room, deepcopy(reward_pos))
    start_room, start_pos = env.agent_room, env.agent_position
    # reward_room, reward_pos = env.reward_room, env.reward_position
    reward_obs = env.get_observation(reward_room, reward_pos)
    reward_global_pos = env.room_centers[reward_room] + reward_pos

    # logging.critical('[traj{}] initial position : room {}, xy {}; target at room {}, xy {}'.format(traj, env.agent_room, env.agent_position, env.reward_room, env.reward_position))

    # room_range = [0]
    room_range = range(env.n_rooms)

    rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in room_range], axis=0)
    # print(rooms)
    x_room = np.linspace(-env.scale, env.scale, resolution)
    # print(x_room)
    y_room = np.linspace(-env.scale, env.scale, resolution)
    xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])
    x = xy_room[:, 0]
    y = xy_room[:, 1]
    x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
    y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
    xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

    # x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
    # y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)

    x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in room_range], axis=0)
    y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in room_range], axis=0)
    xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    env.set_reward_position(reward_room, deepcopy(reward_pos))
    plot_env.set_reward_position(reward_room, deepcopy(reward_pos))
    # print('just before render template', env.reward_room)
    # print('just before render template', env.reward_position)
    ax = plot_env.render_template(ax_to_use=ax)

    for room, pos, global_pos in zip(rooms, xy_local, xy_global):
        env.reset()
        # env.reward_room, env.reward_position = deepcopy(reward_room), deepcopy(reward_pos)

        env.set_reward_position(reward_room, deepcopy(reward_pos))
        plot_env.set_reward_position(reward_room, deepcopy(reward_pos))

        env.set_agent_position(room, pos)

        obs = env.get_observation(room, pos)

        full_obs = {'observation': obs, 'desired_goal': deepcopy(reward_obs), 'achieved_goal': obs}
        action, _ = model.predict(full_obs)

        obs, reward, done, info = env.step(action)

        # Do the correction by environment, otherwise makes the plot more complicated than necessary
        action = info['rectified_action']#.detach().cpu().numpy()
        # ax.arrow(*global_pos, *action, width=.005, color='r', alpha=.9, zorder=10, head_width=.05)
        # ax.arrow(*global_pos, *action, width=.005, color=(global_pos[0] / (6.*env.scale)+.5, global_pos[1]/ (6.*env.scale)+.5, 0) , alpha=.9, zorder=10, head_width=.05)
        r = .5
        ax.arrow(*global_pos, *action, width=.005, color=hsv_to_rgb((global_pos[0] / (6.*env.scale)+.5, 1., (1.-r)*(global_pos[1]/ (6.*env.scale)+.5)+r )) , alpha=.9, zorder=10, head_width=.05)


    plt.savefig('out/tests/policy/{}/frame_{}.png'.format(env_name, frame_idx))
    plt.close()
    frame_idx += 1

# for traj in range(1):
#     resolution = 6
#     _ = env.reset()
#     start_room, start_pos = env.agent_room, env.agent_position
#     reward_room, reward_pos = env.reward_room, env.reward_position
#     reward_obs = env.get_observation(reward_room, reward_pos)
#     reward_global_pos = env.room_centers[reward_room] + reward_pos
#
#     logging.critical('[traj{}] initial position : room {}, xy {}; target at room {}, xy {}'.format(traj, env.agent_room, env.agent_position, env.reward_room, env.reward_position))
#
#     # room_range = [0]
#     room_range = range(env.n_rooms)
#
#     rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in room_range], axis=0)
#     # print(rooms)
#     x_room = np.linspace(-env.scale, env.scale, resolution)
#     # print(x_room)
#     y_room = np.linspace(-env.scale, env.scale, resolution)
#     xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])
#     x = xy_room[:, 0]
#     y = xy_room[:, 1]
#     x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
#     y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
#     xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)
#
#     # x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
#     # y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)
#
#     x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in room_range], axis=0)
#     y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in room_range], axis=0)
#     xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)
#
#
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax = env.render_template(ax_to_use=ax)
#
#     for room, pos, global_pos in zip(rooms, xy_local, xy_global):
#         env.reset()
#         env.reward_room, env.reward_position = reward_room, reward_pos
#         env.set_agent_position(room, pos)
#
#         obs = env.get_observation(room, pos)
#
#         full_obs = {'observation': obs, 'desired_goal': deepcopy(reward_obs), 'achieved_goal': obs}
#         action, _ = model.predict(full_obs)
#
#         obs, reward, done, info = env.step(action)
#
#         # Do the correction by environment, otherwise makes the plot more complicated than necessary
#         action = info['rectified_action']#.detach().cpu().numpy()
#         # ax.arrow(*global_pos, *action, width=.005, color='r', alpha=.9, zorder=10, head_width=.05)
#         # ax.arrow(*global_pos, *action, width=.005, color=(global_pos[0] / (6.*env.scale)+.5, global_pos[1]/ (6.*env.scale)+.5, 0) , alpha=.9, zorder=10, head_width=.05)
#         r = .5
#         ax.arrow(*global_pos, *action, width=.005, color=hsv_to_rgb((global_pos[0] / (6.*env.scale)+.5, 1., (1.-r)*(global_pos[1]/ (6.*env.scale)+.5)+r )) , alpha=.9, zorder=10, head_width=.05)
#
#
#     plt.savefig('out/tests/policy/{}/random.pdf'.format(env_name))
