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

from SensorFusion.rl_related.helpers_RL import *

register(
  id='SnakePath-v0',
  entry_point='environment:FixedRewardWorld')

register(
  id='SnakePath-v1',
  entry_point='environment:FixedRewardPreprocessedWorld')

from stable_baselines3 import PPO, TD3, SAC, DDPG

methods_dict = {'TD3': TD3}

# for env_name, load_from in zip(["SnakePath-v1"], ['out/SnakePath_Default/end_to_end/default/'])
# for env_name, load_from, details_name in zip(["SnakePath-v0"], [None], ['raw']):
for env_name, load_from, use_recurrence, details_name in zip(["SnakePath-v1"], ['out/SnakePath_Default/end_to_end/default/'], [False], ['non_rec']):
# for env_name, load_from, use_recurrence, details_name in zip(["SnakePath-v1"], ['out/SnakePath_Default/end_to_end/default/'], [True], ['rec']):
    method = 'TD3'
    SEED=0
    EVAL_EVERY=5000
    TENSORBOARD_LOG_FOLDER = "./logs/{}_{}/{}/seed{}/tensorboard/".format(env_name, details_name, method, SEED)
    os.makedirs(TENSORBOARD_LOG_FOLDER, exist_ok=True)


    im_availability = .5
    env = gym.make(env_name, seed=SEED, epoch_len=20, im_availability=im_availability, corrupt_frac=.5, use_recurrence=use_recurrence,
                    load_preprocessor_from=load_from)
    env = Monitor(env, TENSORBOARD_LOG_FOLDER)

    eval_env = gym.make(env_name, seed=SEED, epoch_len=20, im_availability=im_availability, corrupt_frac=.5, use_recurrence=use_recurrence,
                    load_preprocessor_from=load_from)
    eval_env = Monitor(eval_env, TENSORBOARD_LOG_FOLDER)


    fig_name_prefix = 'trajectory_figures_'

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=EVAL_EVERY, save_path= "./logs/{}/{}_{}/seed{}/saves/".format(env_name, details_name, method, SEED))
    policy_eval_callback = PolicyEvaluationCallback(eval_env, save_freq=EVAL_EVERY, fig_name_prefix=fig_name_prefix,)

    model = methods_dict[method](
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.9,
        batch_size=1024,

        policy_kwargs={
            'net_arch': [512, 256, 128, 32],
            },
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_FOLDER,
    )


    model.learn(total_timesteps=20000)

    os.makedirs('out/tests/fixed_reward/{}/{}'.format(env_name, method), exist_ok=True)
    logging.critical('start evaluating')
    for traj in range(10):
        obs = env.reset()
        start_room, start_pos = env.agent_room, env.agent_position
        logging.critical('[traj{}] initial position : room {}, xy {}'.format(traj, env.agent_room, env.agent_position))
        actions = np.zeros((20, 2))
        for t in range(20):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            actions[t] = info['rectified_action']#.detach().cpu().numpy()
            logging.critical('[traj{}] action: {}; rectified action {}, reward {}, done {}'.format(traj, action, info['rectified_action'], reward, done))
            logging.critical('[traj{}] current position : room {}, xy {}'.format(traj, env.agent_room, env.agent_position))
            if done:
                break
        env.plot_trajectory(actions, start_room=start_room, start_pos=start_pos, save_file='out/tests/fixed_reward/{}/{}/traj_{}.pdf'.format(env_name, method, traj))
