from stable_baselines3 import SAC
model = SAC('MlpPolicy', 'Pendulum-v0', tensorboard_log='/tmp/sb3/').learn(10000)
