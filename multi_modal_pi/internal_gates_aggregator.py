import torch as tch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import logging
import gc
from matplotlib.patches import Rectangle, Circle
from policy import RandomPolicy
from itertools import cycle
import json
from tqdm import tqdm


from SensorFusion.src.environment import *
from SensorFusion.src.networks import *
from SensorFusion.src.scipy.stats import linregress



max = lambda x, y: x if x > y else y
min = lambda x, y: x if x < y else y

PASTEL_GREEN = "#8fbf8f"
PASTEL_RED = "#ff8080"
PASTEL_BLUE = "#8080ff"
PASTEL_MAGENTA = "#ff80ff"


def plot_mean_std(ax, data, axis=0, c_line='g', c_fill=PASTEL_GREEN, label=None, log_yscale=False):
    if not log_yscale:
        mean =  data.mean(axis=axis)
        std = data.std(axis=axis)
        low = mean - std
        high = mean + std
    else:
        ax.set_yscale('log')
        log_mean = np.log(data).mean(axis=axis)
        log_std = np.log(data).std(axis=axis)
        mean = np.exp(log_mean)
        low = np.exp(log_mean-log_std)
        high = np.exp(log_mean+log_std)

    x = range(mean.shape[0])

    if label is None:
        ax.plot(x, mean, c=c_line)
    else:
        ax.plot(x, mean, c=c_line, label=label)

    ax.fill_between(x, low, high, color=c_fill, alpha=.7, zorder=1)


def offline_study_internal_gates(map='SnakePath', layout='Default', exp_group='end_to_end', protocol='default', batch_size=64, n_trajs=512, epoch_len=50, step_size=.5, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, reimplementation_type='Small'):
    n_trajs_to_plot = 10
    n_neurons_to_plot = 10
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    path = 'out/{}_{}/{}/{}/'.format(map, layout, exp_group, protocol)

    with open(path+'full_params.json') as f:
        all_params = json.load(f)
        env = FixedRewardWorld(**all_params['env_params'], seed=777)
        net = BigReimplementationPathIntegrator(**all_params['net_params']['options'])

    os.makedirs(path+'gatings_offline_study/', exist_ok=True)

    rec_type = net.recurrence_type

    actions = step_size * np.random.randn(n_trajs, epoch_len, 2)
    rooms, positions, actions = env.static_replay(actions)
    cumulated_actions = np.cumsum(actions, axis=1)

    reset_every = int(1/im_availability)
    ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((batch_size, 1)).float()
    ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

    # Now, for images to perturb, choose between "corruption" and "drop"
    corrupt = tch.where(tch.bernoulli(corruption_rate * tch.ones(batch_size, epoch_len+1)).byte(), ims_to_perturb, tch.zeros(batch_size, epoch_len+1)).bool()
    drop = tch.logical_and(ims_to_perturb, tch.logical_not(corrupt))
    mask = (1.-drop.float()).unsqueeze(-1).repeat(1, 1, net.representation_size).float().to(net.device)

    time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[1]+1)
    cmap = plt.get_cmap('jet')
    colors = cmap(time_based_norm(range(epoch_len+1)))

    marker = '*'
    marker_true = '+'

    # Just to ensure that reimplementation works correctly
    all_errors = np.zeros((n_trajs*n_seeds, epoch_len))
    # Use same name for GRU and LSTM, despite not being strictly identical
    all_resets = np.zeros((n_trajs*n_seeds, epoch_len, net.representation_size))
    all_inputs = np.zeros((n_trajs*n_seeds, epoch_len, net.representation_size))

    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            print('Starting seed {}'.format(start_seed+seed))
            # env = World(**all_params['env_params'], seed=start_seed+seed)
            env = FixedRewardWorld(**all_params['env_params'], seed=start_seed+seed)
            params = all_params['net_params']['options']
            params['use_reimplementation'] = True

            print(rec_type, path+'seed{}/best_net.tch'.format(start_seed+seed))

            net = BigReimplementationPathIntegrator(**params)

            net.load(path+'seed{}/best_net.tch'.format(start_seed+seed))

            # Then, the noiseless ones
            for batch_idx in range(n_trajs//batch_size):
                images = env.get_images(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size], positions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                actions_encodings = net.get_z_encoding(tch.from_numpy(actions[batch_idx*batch_size:(batch_idx+1)*batch_size]).view(batch_size * (epoch_len), 2).float().to(net.device))
                actions_encodings = actions_encodings.view(batch_size, (epoch_len), -1)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                if rec_type == 'GRU':
                    outputs, _, resetgates, inputgates, _ = net.do_path_integration(representations, actions_encodings, return_all=True)
                elif rec_type == 'LSTM':
                    outputs, _, inputgates, resetgates, _, _ = net.do_path_integration(representations, actions_encodings, return_all=True)

                outputs = outputs.detach().cpu().numpy()
                inputgates = inputgates.detach().cpu().numpy()
                resetgates = resetgates.detach().cpu().numpy()

                # all_errors[seed*n_trajs+batch_idx*batch_size:seed*n_trajs+(batch_idx+1)*batch_size] = outputs[:,:,0]
                all_errors[seed*n_trajs+batch_idx*batch_size:seed*n_trajs+(batch_idx+1)*batch_size] = np.sqrt(((outputs-cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))
                all_resets[seed*n_trajs+batch_idx*batch_size:seed*n_trajs+(batch_idx+1)*batch_size] = resetgates
                all_inputs[seed*n_trajs+batch_idx*batch_size:seed*n_trajs+(batch_idx+1)*batch_size] = inputgates



        # First, plot some examples:
        selected_seeds = np.random.randint(n_seeds, size=n_neurons_to_plot)
        selected_neurons = np.random.randint(net.representation_size, size=n_neurons_to_plot)

        selected_neurons_resets = np.zeros((n_neurons_to_plot, n_trajs, epoch_len))
        selected_neurons_inputs = np.zeros((n_neurons_to_plot, n_trajs, epoch_len))
        for idx in range(n_neurons_to_plot):
            selected_neurons_resets[idx] = all_resets[selected_seeds[idx]*n_trajs:(selected_seeds[idx]+1)*n_trajs, :, selected_neurons[idx]]
            selected_neurons_inputs[idx] = all_inputs[selected_seeds[idx]*n_trajs:(selected_seeds[idx]+1)*n_trajs, :, selected_neurons[idx]]

        all_resets = np.transpose(all_resets, axes=(0,2,1)).reshape((-1, epoch_len))
        all_inputs = np.transpose(all_inputs, axes=(0,2,1)).reshape((-1, epoch_len))

        reorder = np.random.permutation(all_resets.shape[0])[:n_trajs_to_plot]
        selected_resets = all_resets[reorder]
        selected_inputs = all_inputs[reorder]

        for idx in range(n_trajs_to_plot):
            plt.figure()
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    plt.axvline(x=t, ls='--', c='k')
            plt.plot(selected_inputs[idx], label=r'Input gate')
            plt.savefig(path+'gatings_offline_study/input_gate_traj{}.pdf'.format(idx))

            plt.figure()
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    plt.axvline(x=t, ls='--', c='k')
            plt.plot(selected_resets[idx], label=r'Reset gate')
            plt.savefig(path+'gatings_offline_study/reset_gate_traj{}.pdf'.format(idx))
            plt.close('all')

        for idx in range(n_neurons_to_plot):
            fig, ax = plt.subplots()
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    plt.axvline(x=t, ls='--', c='k')
            plot_mean_std(ax, selected_neurons_inputs[idx])
            plt.savefig(path+'gatings_offline_study/input_gate_averaged_neuron_{}.pdf'.format(idx))

            fig, ax = plt.subplots()
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    plt.axvline(x=t, ls='--', c='k')
            plot_mean_std(ax, selected_neurons_resets[idx])
            plt.savefig(path+'gatings_offline_study/reset_gate_averaged_neuron_{}.pdf'.format(idx))
            plt.close('all')

        # Then, plot averages
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax = axes[0]
        plot_mean_std(ax, all_resets)
        ax.set_title('Reset gates', fontsize=20)

        ax = axes[1]
        plot_mean_std(ax, all_inputs)
        ax.set_title('Input gates', fontsize=20)

        fig.savefig(path+'gatings_offline_study/averaged_gatings.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for t in range(1, epoch_len):
            if ims_to_perturb[0, t+1] == 0:
                ax.axvline(x=t, ls='--', c='k')
        plot_mean_std(ax, all_errors)
        plt.savefig(path+'gatings_offline_study/__error_sanity_check.pdf')



if __name__ == '__main__':
    offline_study_internal_gates(map='SnakePath', layout='Default', exp_group='long_experiment', protocol='offshelf_LSTM/pretrained/error_0.0_avail_0.2', batch_size=64, n_trajs=512, epoch_len=50, step_size=.5, start_seed=0, n_seeds=8, im_availability=.1, corruption_rate=.5)
