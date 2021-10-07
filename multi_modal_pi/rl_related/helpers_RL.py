import gym
import torch as tch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO, TD3, SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.logger import Figure
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        self._features_dim = 512 * 2 # No beating around the bush, can make it variable later
        # self.common_layers = nn.Sequential(
        #     nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, 256), nn.ReLU(),
        # )

    def forward(self, observations) -> tch.Tensor:
        inputs = tch.cat([observations['observation'], observations['desired_goal']], dim=1)
        # logging.critical([inputs.shape, self.common_layers(inputs).shape])
        return inputs
        # return self.common_layers(inputs)

# Callback for evaluating the policy (plot 10 example trajectories)
class PolicyEvaluationCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0, save_freq=10000, fig_name_prefix='figure_'):
        super(PolicyEvaluationCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.fig_name_prefix = fig_name_prefix
        self.eval_env = eval_env

    def _on_step(self):
        # Plot values (here a random variable)
        if self.num_timesteps == 0 or self.num_timesteps % self.save_freq != 0:
            return True

        print('working on the policyeval')
        env = self.eval_env

        for traj in range(20):
            obs = env.reset()
            start_room, start_pos = env.agent_room, env.agent_position
            reward_room, reward_pos = env.reward_room, env.reward_position
            actions = np.zeros((20, 2))
            for t in range(20):
                action, _states = self.model.predict(obs)
                obs, reward, done, info = env.step(action)
                actions[t] = info['rectified_action']#.detach().cpu().numpy()
                if done:
                    break
            # logging.critical('exited inner loop')
            figure = env.plot_trajectory(actions, start_room=start_room, start_pos=start_pos, reward_room=reward_room, reward_pos=reward_pos, save_file=None, return_fig=True)
            # self.logger.record(self.save_path + 'trajectories/{}/{}.pdf'.format(self.num_timesteps, traj), Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            # self.logger.record(self.fig_name_prefix + 'step_{}_traj_{}.pdf'.format(self.num_timesteps, traj), Figure(figure, close=False), exclude=("stdout", "log", "json", "csv"))
            self.logger.record(self.fig_name_prefix + 'trajectories_latest_{}.pdf'.format(traj), Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close('all')

        return True
