from gym.envs.registration import register
import gym
import os
import numpy as np
import logging

register(
  id='SnakePath-v0',
  entry_point='environment:FixedRewardWorld')

register(
  id='SnakePath-v1',
  entry_point='environment:FixedRewardPreprocessedWorld')

from stable_baselines3 import PPO, TD3, SAC, DDPG

# env = gym.make("SnakePath-v0", seed=0)
for env_name in ["SnakePath-v0", "SnakePath-v1"]:
    env = gym.make(env_name, seed=0)

    # Works fine, so stick with it
    # model = PPO("CustomPolicy", env, verbose=1, gamma=0.95)
    # model = PPO("MlpPolicy", env, verbose=1, gamma=0.9)
    # model.learn(total_timesteps=200000) # good for Default, somehow ok for TopRight?

    # Works even better
    # model = DDPG("MlpPolicy", env, verbose=1, gamma=0.9, batch_size=1024)
    # model.learn(total_timesteps=10000)

    # Very similar, albeit a bit slower?
    model = TD3("MlpPolicy", env, verbose=1, gamma=0.9, batch_size=1024)
    model.learn(total_timesteps=20000)

    os.makedirs('out/tests/baseline/{}'.format(env_name), exist_ok=True)
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
        env.plot_trajectory(actions, start_room=start_room, start_pos=start_pos, save_file='out/tests/baseline/{}/traj_{}.pdf'.format(env_name, traj))
