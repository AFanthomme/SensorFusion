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
from environment import *
from networks import *
# from reimplementations import ReimplementationPathIntegrator, BigReimplementationPathIntegrator
from scipy.stats import linregress
from tqdm import tqdm
from networks import network_register
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

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

def representation_curriculum_comparison(map='SnakePath', layout='Default', n_seeds=8, resolution=20):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    path = 'out/{}_{}/representations/'.format(map, layout)
    tmp = ['{}_from_scratch/'.format(b) for b in ['adjacent', 'full_space']] + ['full_space_from_adjacent/']
    folder_labels = ["'Physically' allowed", "Whole space from scratch", "Whole space from physically allowed"]

    folders = [path + f for f in tmp]

    with open(folders[0] + 'full_params.json', 'r') as f:
        env_params = json.load(f)['env_params']
        env = World(**env_params, seed=777)

    n_rooms = env.n_rooms

    x_room = np.linspace(-env.scale, env.scale, resolution)
    y_room = np.linspace(-env.scale, env.scale, resolution)
    xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])

    if env.map_name in ['BigRoom']:
        rooms = np.zeros(resolution**2).astype(int)
        x_global = x_room
        y_global = y_room
        xy_global = xy_room
        xy_local = xy_room

        start_pos_list = [
            (np.array([0]), np.array([0., 0.])),
            (np.array([0]), np.array([-0.5*env.scale, 0.])),
            (np.array([0]), np.array([-1*env.scale, -1.*env.scale])),
            (np.array([0]), np.array([-.4*env.scale, -.4*env.scale])),
        ]
        start_labels = ['center', 'offset', 'corner', 'diagonal']

    elif env.map_name in ['DonutPath', 'SnakePath']:
        rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in range(env.n_rooms)], axis=0)
        logging.critical(np.bincount(rooms))
        x = xy_room[:, 0]
        y = xy_room[:, 1]

        x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
        y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)
        xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

        x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
        y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
        xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

        if env.map_name == 'SnakePath':
            start_pos_list = [
                (np.array([4]), np.array([0., 0.])),
                (np.array([0]), np.array([0, 0.])),
                (np.array([1]), np.array([-.8*env.scale, 0.5*env.scale])),
            ]
            start_labels = ['center', 'bot_left', 'offset']
        elif env.map_name == 'DonutPath':
            start_pos_list = [
                (np.array([0]), np.array([0, 0.])),
                (np.array([1]), np.array([-1*env.scale, 0.5*env.scale])),
            ]
            start_labels = ['bot_left', 'offset']

    elif env.map_name == 'DoubleDonut':
        rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in range(env.n_rooms)], axis=0)
        logging.critical(np.bincount(rooms))
        x = xy_room[:, 0]
        y = xy_room[:, 1]

        x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
        y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)
        xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

        x_figure = np.concatenate([x+env.room_centers[room_idx][0] + 7 * env.scale * env.room_centers[room_idx][2] for room_idx in range(env.n_rooms)], axis=0)
        xy_figure = np.concatenate([x_figure.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

        x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
        y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
        xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

        start_pos_list = [
            (np.array([0]), np.array([0, 0.])),
            (np.array([1]), np.array([-1*env.scale, 0.5*env.scale])),
            (np.array([3]), np.array([0, 0.])),
            (np.array([13]), np.array([0, 0.])),
        ]
        start_labels = ['bottom_left', 'bottom_left_offset', 'middle', 'top_right']

    else:
        raise RuntimeError('Undefined environment for curriculum aggregator')

    all_images = env.get_images(rooms, xy_local)

    for start_label, start_pos_tup in zip(start_labels, start_pos_list):
        start_room, start_pos = start_pos_tup
        delta_r = xy_global - env.room_centers[start_room, :2] - start_pos

        averaged_errors = []
        for folder_idx, folder in enumerate(folders):
            errors_blob = np.zeros((n_seeds, xy_global.shape[0]))
            with tch.set_grad_enabled(False):
                for seed in range(n_seeds):
                    with open(folder + 'full_params.json', 'r') as f:
                        net_params = json.load(f)['net_params']
                        net_params['options']['seed'] = seed
                        net = network_register[net_params['net_name']](**net_params['options'])

                    net.load(folder + 'seed{}/best_net.tch'.format(seed))

                    start_rep = net.get_representation(env.get_images(start_room, start_pos))
                    all_reps = net.get_representation(all_images)

                    dists = net.backward_model(tch.cat([start_rep.reshape([1, -1]) for _ in range(all_reps.shape[0])], dim=0), all_reps).detach().cpu().numpy()
                    errors_blob[seed] = np.sqrt(((dists-delta_r)**2).sum(axis=-1))
            averaged_errors.append(np.mean(errors_blob, axis=0))


        seismic = plt.get_cmap('seismic')
        reds = plt.get_cmap('Reds')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        errors = np.stack(averaged_errors, axis=0)
        log_errors = np.log10(errors)
        norm_log = matplotlib.colors.Normalize(vmin=log_errors.min(), vmax=log_errors.max())

        for folder_idx, f in enumerate(folder_labels):
            ax = axes[folder_idx]
            ax = env.render_template(ax_to_use=ax)
            patch = Circle((env.room_centers[start_room,0] + start_pos[0], env.room_centers[start_room,1] + start_pos[1]), .1 * env.scale, linewidth=1, edgecolor='k', facecolor=[0,0,0,.2])
            ax.add_patch(patch)
            ax.set_title('Reconstruction error\n' + folder_labels[folder_idx])
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm_log(log_errors[folder_idx].flatten())), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm_log, orientation='vertical')
            fig.add_axes(ax_cb)

        fig.tight_layout()
        fig.savefig(path + 'log_errors_comparison_{}.pdf'.format(start_label))
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        relative_log_errors = log_errors - .5 * np.log10((dists**2).sum(axis=-1))
        norm_relative_log = matplotlib.colors.Normalize(vmin=relative_log_errors.min(), vmax=relative_log_errors.max())

        for folder_idx, f in enumerate(folder_labels):
            ax = axes[folder_idx]
            ax = env.render_template(ax_to_use=ax)
            patch = Circle((env.room_centers[start_room,0] + start_pos[0], env.room_centers[start_room,1] + start_pos[1]), .1 * env.scale, linewidth=1, edgecolor='k', facecolor=[0,0,0,.2])
            ax.add_patch(patch)
            ax.set_title('Reconstruction error\n' + folder_labels[folder_idx])
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm_relative_log(relative_log_errors[folder_idx].flatten())), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm_relative_log, orientation='vertical')
            fig.add_axes(ax_cb)

        fig.tight_layout()
        fig.savefig(path + 'relative_log_errors_comparison_{}.pdf'.format(start_label))
        plt.close(fig)


def offline_gating_compare_training_noise(map='SnakePath', layout='Default', common_folder_root='annealed_several_noises_{}', suffixes=['000/', '001/', '0025/', '005/', '0075/', '01/'], noise_levels = [0., .01, .025, .05, .075, .1], start_seed=0, n_seeds=8, resolution=20):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    n_noise_levels = len(noise_levels)

    perturbation_levels = np.array([0., 0.1, .25, .5, .75, 1.])
    n_repeats_per_level = 10

    path = 'out/{}_{}/end_to_end/'.format(map, layout)
    path = path + common_folder_root

    all_folders = [path.format(suffix) for suffix in suffixes]

    with open(all_folders[0]+'full_params.json') as f:
        all_params = json.load(f)
        env = World(**all_params['env_params'], seed=777)

    # all_log_ps_unperturbed = np.zeros((n_seeds, env.n_rooms * (resolution**2)))
    all_log_ps = np.zeros((n_noise_levels, n_seeds, env.n_rooms * (resolution**2)))
    log_ps_for_second_panel = np.zeros((6, n_repeats_per_level, n_seeds, env.n_rooms * (resolution**2)))

    x_room = np.linspace(-env.scale, env.scale, resolution)
    y_room = np.linspace(-env.scale, env.scale, resolution)
    xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])
    rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in range(env.n_rooms)], axis=0)
    x = xy_room[:, 0]
    y = xy_room[:, 1]

    x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
    y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
    xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

    x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
    y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)
    xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

    all_images = env.get_images(rooms, xy_local)

    with tch.set_grad_enabled(False):
        for noise_idx, f, noise in zip(range(n_noise_levels), all_folders, noise_levels):
            for seed in range(n_seeds):
                print('Starting seed {}'.format(start_seed+seed))
                env = World(**all_params['env_params'], seed=start_seed+seed)
                net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
                net.load_state_dict(tch.load(f+'seed{}/best_net.tch'.format(start_seed+seed)))
                all_reps = net.get_representation(all_images)

                log_p = net.gating_module(tch.cat([tch.zeros_like(all_reps), all_reps], dim=1))[:, 0]
                all_log_ps[noise_idx, seed] = log_p.cpu().numpy()

                # For the most noisy, study as a function of perturbation level
                if noise_idx == (n_noise_levels-1):
                    # print(perturbation_levels)
                    for perturb_level_idx, perturb_level in enumerate(perturbation_levels):
                        for perturb_repeat in tqdm(range(n_repeats_per_level)):
                            all_reps_loc = all_reps.clone()
                            if perturb_level > 0:
                                n_to_perturb = int(net.representation_size*perturb_level)
                                subset = np.random.permutation(range(net.representation_size))[:n_to_perturb]
                                all_reps_loc[:, subset] = all_reps_loc[:, subset[np.random.permutation(range(n_to_perturb))]]
                            log_p = net.gating_module(tch.cat([tch.zeros_like(all_reps_loc), all_reps_loc], dim=1))[:, 0]
                            log_ps_for_second_panel[perturb_level_idx, perturb_repeat, seed] = log_p.cpu().numpy()

    # For darkcenter, remove the dark room
    if map == 'SnakePath' and layout == 'DarkCenter':
        all_log_ps = all_log_ps[:, :, :, rooms!=4]
        log_ps_for_second_panel = log_ps_for_second_panel[:, :, :, rooms!=4]

    if map == 'DoubleDonut' and layout == 'Ambiguous':
        all_log_ps = all_log_ps[:, :, :, np.logical_and(rooms!=7, rooms!=11)]
        log_ps_for_second_panel = log_ps_for_second_panel[:, :, :, np.logical_and(rooms!=7, rooms!=11)]

    np.save(path.format('all_log_ps.npy'), all_log_ps)
    np.save(path.format('log_ps_for_second_panel.npy'), log_ps_for_second_panel)

    colors = matplotlib.cm.get_cmap('viridis')(np.linspace(0,1, n_noise_levels))

    # fig, ax = plt.subplots()
    # for noise_idx, f, noise, c in zip(range(n_noise_levels), all_folders, noise_levels, colors):
    #     sns.histplot(all_log_ps[noise_idx].flatten(), ax=ax, label=r'PI training noise level : {}%'.format(noise), color=c)
    # ax.legend()
    # fig.savefig(path.format('gating_distributions_training_noise.pdf'))


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for noise_idx, f, noise, c in zip(range(n_noise_levels), all_folders, noise_levels, colors):
        sns.ecdfplot(all_log_ps[noise_idx].flatten(), ax=ax, label=r'PI training noise level : {}%'.format(noise))
    ax.legend()
    ax.set_xlabel(r"Logarithm of the reset gate value")
    ax.set_ylabel(r"Cumulative distribution function")

    ax = axes[1]
    for perturb_level_idx, perturb_level, c in zip(range(len(perturbation_levels)), perturbation_levels, colors):
        sns.ecdfplot(log_ps_for_second_panel[perturb_level_idx].flatten(), ax=ax, label=r'Representation perturbation level : {}%'.format(perturb_level))
    ax.legend()
    ax.set_xlabel(r"Logarithm of the reset gate value")
    ax.set_ylabel(r"Cumulative distribution function")

    fig.tight_layout()
    fig.savefig(path.format('gating_cdfs_training_noise.pdf'))

def offline_gating_plot(map='SnakePath', layout='Default', protocol='default', start_seed=0, n_seeds=2, resolution=50):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    seismic = plt.get_cmap('seismic')

    path = 'out/{}_{}/end_to_end/'.format(map, layout)
    path = path + protocol + '/'
    os.makedirs(path+'offline_gating_plot/', exist_ok=True)

    with open(path+'full_params.json') as f:
        all_params = json.load(f)
        env = World(**all_params['env_params'], seed=777)

    x_room = np.linspace(-env.scale, env.scale, resolution)
    y_room = np.linspace(-env.scale, env.scale, resolution)
    xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])
    rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in range(env.n_rooms)], axis=0)
    x = xy_room[:, 0]
    y = xy_room[:, 1]

    x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
    y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
    xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

    x_global = np.concatenate([x+env.room_centers[room_idx][0] for room_idx in range(env.n_rooms)], axis=0)
    y_global = np.concatenate([y+env.room_centers[room_idx][1] for room_idx in range(env.n_rooms)], axis=0)
    xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

    all_images = env.get_images(rooms, xy_local)
    all_log_ps = np.zeros((n_seeds, env.n_rooms*(resolution**2)))

    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            print('Starting seed {}'.format(start_seed+seed))
            env = World(**all_params['env_params'], seed=start_seed+seed)
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
            net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(start_seed+seed)))
            all_reps = net.get_representation(all_images)

            log_p = net.gating_module(tch.cat([tch.zeros_like(all_reps), all_reps], dim=1))[:, 0].detach().cpu().numpy()
            norm = matplotlib.colors.Normalize(vmin=log_p.min(), vmax=log_p.max())
            all_log_ps[seed] = log_p

            if map != 'DoubleDonut':
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            elif map == 'DoubleDonut':
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            ax = env.render_template(ax_to_use=ax)
            ax.set_title(r'Value of the gating ($\log_e$) \\')
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm(log_p)), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
            fig.add_axes(ax_cb)

            fig.tight_layout()
            fig.savefig(path + 'offline_gating_plot/offline_gating_plot_seed_{}.pdf'.format(start_seed+seed))
            plt.close(fig)

        log_p = all_log_ps.mean(axis=0)
        norm = matplotlib.colors.Normalize(vmin=log_p.min(), vmax=log_p.max())
        if map != 'DoubleDonut':
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        elif map == 'DoubleDonut':
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax = env.render_template(ax_to_use=ax)
        ax.set_title(r'Value of the gating ($\log_e$) \\')
        ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm(log_p)), s=64000/(resolution**2), rasterized=True, zorder=-5)
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
        fig.add_axes(ax_cb)

        fig.tight_layout()
        fig.savefig(path + 'offline_gating_plot/offline_gating_plot_averaged.pdf')
        plt.close(fig)








def offline_make_deliberate_trajectory_figures(map='SnakePath', layout='Default', protocol='default', batch_size=4, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, noise=0., resetting_type='fixed', figure_layout='one_row'):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    path = 'out/{}_{}/end_to_end/{}/'.format(map, layout, protocol)
    print(path)

    with open(path+'full_params.json') as f:
        all_params = json.load(f)
        env = World(**all_params['env_params'], seed=777)
        net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])

        deliberate_actions = np.array([meaningful_trajectories[env.map_name]]*batch_size) * env.scale
        epoch_len = deliberate_actions.shape[1]
        rooms, positions, _ = env.static_replay(deliberate_actions, start_rooms=np.zeros(batch_size, dtype=int), start_pos=np.zeros((batch_size, 2)))

        perturbed_positions = np.clip(positions+.2*env.scale*np.random.uniform(-1, 1, size=positions.shape), -env.scale, env.scale)
        global_positions = perturbed_positions + env.room_centers[rooms.astype(int)][:,:,:2]
        actions = global_positions[:, 1:] - global_positions[:, :-1]

        actions[0] = deliberate_actions[0]
        epoch_len = actions.shape[1]
        start_rooms = np.zeros(batch_size)
        start_pos = np.zeros((batch_size, 2))

        rooms, positions, actions = env.static_replay(actions, start_rooms=start_rooms, start_pos=start_pos)
        cumulated_actions = np.cumsum(actions, axis=1)
        images = env.get_images(rooms, positions) #retinal states, (bs, T+1=2, ret_res**2, 3)

        if resetting_type == 'fixed':
            reset_every = int(1/im_availability)
            ims_to_perturb = ((tch.tensor(range(epoch_len+1))).fmod(reset_every)!=0).unsqueeze(0).repeat((batch_size, 1)).float()
        elif resetting_type == 'random':
            ims_to_perturb =  tch.bernoulli((1.-im_availability) * tch.ones(batch_size, epoch_len+1))

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

    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            print('Starting seed {}'.format(start_seed+seed))
            env = World(**all_params['env_params'], seed=start_seed+seed)
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
            net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(start_seed+seed)))

            outputs = np.zeros((batch_size, epoch_len, 2))
            gatings = np.zeros((batch_size, epoch_len))
            representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
            actions_encodings =  net.get_z_encoding(tch.from_numpy(actions + noise*np.random.randn(*actions.shape)).view(batch_size * (epoch_len), 2).float().to(net.device)).view(batch_size, (epoch_len), -1)
            representations = mask * representations
            tmp = representations[corrupt]
            tmp = tmp[:, tch.randperm(tmp.shape[1])]
            representations[corrupt] = tmp

            outputs, gatings, _ = net.do_path_integration(representations, actions_encodings)
            outputs = outputs.detach().cpu().numpy()
            gatings = gatings.detach().cpu().numpy()

            for b in range(batch_size):
                true_global_pos  = np.zeros((epoch_len+1, 2))
                true_global_pos[0] = env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                true_global_pos[1:] = cumulated_actions[b] + env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate

                global_pos  = np.zeros((epoch_len+1, 2))
                global_pos[0] = env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                global_pos[1:] = outputs[b] + env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                errors = np.sqrt(((outputs-cumulated_actions)**2).sum(axis=-1))


                if figure_layout == 'single_row':
                    fig = plt.figure(tight_layout=True, figsize=(10, 5))
                elif figure_layout == 'two_rows':
                    fig = plt.figure(tight_layout=True, figsize=(10, 10))

                gs = matplotlib.gridspec.GridSpec(2, 2)

                if figure_layout == 'single_row':
                    ax = fig.add_subplot(gs[:, 0])
                elif figure_layout == 'two_rows':
                    ax = fig.add_subplot(gs[0, :])
                ax = env.render_template(ax_to_use=ax)
                for t in range(epoch_len - 1):
                    ax.scatter(global_pos[t,0], global_pos[t,1], c=colors[t], marker=marker, alpha=.5, zorder=-5)
                    ax.scatter(true_global_pos[t,0], true_global_pos[t,1], c=colors[t], marker=marker_true, alpha=.5, zorder=-5)
                    if ims_to_perturb[b, t] == 0:
                        ax.scatter(true_global_pos[t,0], true_global_pos[t,1], edgecolors='k', s=80, facecolors='none', alpha=.5, zorder=.5)

                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=time_based_norm, orientation='vertical')
                fig.add_axes(ax_cb)
                ax.set_title('Recovered positions')
                if figure_layout == 'single_row':
                    ax = fig.add_subplot(gs[0, 1])
                elif figure_layout == 'two_rows':
                    ax = fig.add_subplot(gs[1, 0])
                ax.plot(gatings[b, :epoch_len, 0])
                ax.axhline(y=0, c='k')
                for t in range(epoch_len):
                    if ims_to_perturb[b, t+1] == 0:
                        ax.axvline(x=t, ls='--', c='k')
                ax.set_title('Value of the gating')

                ax = fig.add_subplot(gs[1, 1])
                ax.semilogy(errors[b, :epoch_len])
                ax.axhline(y=0, c='k')
                for t in range(epoch_len):
                    if ims_to_perturb[b, t+1] == 0:
                        ax.axvline(x=t, ls='--', c='k')
                ax.set_title('Value of the error')

                os.makedirs(path+'seed{}/offline_deliberate_trajectory_figures'.format(start_seed+seed), exist_ok=True)
                fig.savefig(path+'seed{}/offline_deliberate_trajectory_figures/im_availability_{}_noise_{}_resetting_{}_traj{}.pdf'.format(start_seed+seed, im_availability, noise, resetting_type, b))
                plt.close(fig)


def make_retraining_comparison(map='SnakePath', layout='Default', protocols=['retrain_reset_only', 'retrain_all', 'default'], start_seed=0, n_seeds=4, n_epochs=4000):
    # Add the titles manually, along with the lettering
    # Meant to have three protocols, but leave it be so we can plot while training
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    all_losses = np.zeros((len(protocols), n_seeds, n_epochs, 6))

    for protocol_idx, protocol in enumerate(protocols):
        path = 'out/{}_{}/retrain_for_PI/{}/'.format(map, layout, protocol)
        for seed in range(n_seeds):
            all_losses[protocol_idx, seed] = np.loadtxt(path+'seed{}/losses.txt'.format(seed+start_seed))[:n_epochs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['r', 'b', 'g', 'm']
    pastel_colors = [PASTEL_RED, PASTEL_BLUE, PASTEL_GREEN, PASTEL_MAGENTA]

    min, max = np.min(all_losses), np.max(all_losses)

    for protocol_idx, protocol in enumerate(protocols):
        ax = axes[protocol_idx]
        losses = all_losses[protocol_idx]
        for loss_idx, c, c_pastel, label in zip(range(4), colors, pastel_colors, [r'Forward', 'Backward', 'Reinference', 'Path Integration']):
            ax.set_ylim(min, max)
            ax.set_ylabel('Logarithm of the loss')
            ax.set_xlabel('Training epoch')
            plot_mean_std(ax, losses[:,:,loss_idx], axis=0, c_line=c, c_fill=c_pastel, label=label, log_yscale=True)
            ax.legend()

    fig.tight_layout()
    fig.savefig('out/{}_{}/retrain_for_PI/losses_comparison.pdf'.format(map, layout)) # TODO: unindent once when last folder done running


def offline_study_dynamic_representation(map='SnakePath', layout='Default', protocol='default', n_seeds=1, n_trajs=128, epoch_len=200, batch_size = 64):
    # Can make long trajectories without risking an out-of-memory

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    path = 'out/{}_{}/end_to_end/{}/'.format(map, layout, protocol)

    reset_every = 5
    corruption_rate = .5

    with open(path+'/full_params.json') as f:
        all_params = json.load(f)
        policy_pars = all_params['policy_params']


    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            os.makedirs(path+'offline_study_dynamic_representation/'.format(seed), exist_ok=True)
            print('Starting seed {}'.format(seed))
            env = World(**all_params['env_params'], seed=seed)
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
            net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(seed)), strict=False)

            if policy_pars['type'] == 'CoherentPolicy':
                policy = CoherentPolicy(step_size=policy_pars['step_size'], coherence=policy_pars['coherence'], seed=seed)
            elif policy_pars['type'] == 'RandomPolicy':
                policy = RandomPolicy(step_size=policy_pars['step_size'], seed=seed)


            all_actions = policy.get_batch_of_actions(batch_size=n_trajs, epoch_len=epoch_len)
            all_rooms = np.zeros((n_trajs, epoch_len))
            all_positions = np.zeros((n_trajs, epoch_len, 2))
            all_visual_representations = np.zeros((n_trajs, epoch_len, 16))
            all_activations = np.zeros((n_trajs, epoch_len, 16)) # Only keep the first 16 neurons since that's al we will be plotting

            for batch_idx in range(n_trajs // batch_size):
                rooms, positions, actions = env.static_replay(all_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                all_rooms[batch_idx*batch_size:(batch_idx+1)*batch_size] = rooms[:, 1:]
                all_positions[batch_idx*batch_size:(batch_idx+1)*batch_size] = positions[:, 1:]

                cumulated_actions = np.cumsum(actions, axis=1)

                images = env.get_images(rooms, positions) #retinal states, (bs, T+1=2, ret_res**2, 3)
                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                all_visual_representations[batch_idx*batch_size:(batch_idx+1)*batch_size] = representations[:, 1:, :16].detach().cpu().numpy()
                z_encodings =  net.get_z_encoding(tch.from_numpy(actions).view(batch_size * (epoch_len), 2).float().to(net.device)).view(batch_size, (epoch_len), -1)

                ims_to_perturb = ((tch.tensor(range(epoch_len+1))+1).fmod(reset_every)!=0).unsqueeze(0).repeat((batch_size, 1)).float()
                # Never mess with anchor point, otherwise whole trajectory is meaningless
                ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

                # Now, for images to perturb, choose between "corruption" and "drop"
                corrupt = tch.where(tch.bernoulli(corruption_rate * tch.ones(batch_size, epoch_len+1)).byte(), ims_to_perturb, tch.zeros(batch_size, epoch_len+1)).bool()
                drop = tch.logical_and(ims_to_perturb, tch.logical_not(corrupt))
                mask = (1.-drop.float()).unsqueeze(-1).repeat(1, 1, net.representation_size).float().to(net.device)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                _, _, internal_states = net.do_path_integration(representations, z_encodings)

                all_activations[batch_idx*batch_size:(batch_idx+1)*batch_size] = internal_states[:, :, :16].detach().cpu().numpy()

            seismic = plt.get_cmap('seismic')
            all_rooms = all_rooms.astype(int)
            print(env.room_centers[all_rooms].shape, all_positions.shape)

            all_global_positions = env.room_centers[all_rooms][:,:,:2].reshape(-1, 2) + all_positions.reshape(-1, 2)


            for neuron in range(16):
                neuron_activations = all_activations[:,:,neuron]
                neuron_visual_representation = all_visual_representations[:,:,neuron]
                norm = matplotlib.colors.Normalize(vmin=min(neuron_activations.min(), neuron_visual_representation.min()), vmax=max(neuron_activations.max(), neuron_visual_representation.max()))

                fig, axes = plt.subplots(2, 1, figsize=(6,6))
                ax = axes[0]
                ax = env.render_template(ax_to_use=ax)
                neuron_colors = seismic(norm(neuron_visual_representation)).reshape((-1, 4))
                ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))

                ax = axes[1]
                ax = env.render_template(ax_to_use=ax)
                neuron_colors = seismic(norm(neuron_activations)).reshape((-1, 4))
                ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))

                fig.savefig(path+'/offline_study_dynamic_representation/seed_{}_neuron_{}.png'.format(seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                plt.close('all')




def offline_study_errors(map='SnakePath', layout='Default', exp_group='end_to_end', protocol='default', batch_size=64, n_trajs=512, epoch_len=50, step_size=.5, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', use_reimplementation=None, collapsed=False):
    base_path = 'out/{}_{}/'.format(map, layout)
    path = 'out/{}_{}/{}/{}/'.format(map, layout, exp_group, protocol)
    figs_path = 'out/{}_{}/{}/figures/{}/'.format(map, layout, exp_group, protocol)

    with open(path+'full_params.json') as f:
        all_params = json.load(f)
        env = FixedRewardWorld(**all_params['env_params'], seed=777)
        net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])

    try:
        initial_actions = np.load('out/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}.npy'.format(epoch_len, step_size, n_trajs, im_availability))
    except Exception as e:
        logging.critical(e)
        initial_actions = step_size * np.random.randn(n_trajs, epoch_len, 2)
        np.save('out/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}.npy'.format(epoch_len, step_size, n_trajs, im_availability), initial_actions)

    try:
        rooms = np.load(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_rooms.npy'.format(epoch_len, step_size, n_trajs, im_availability))
        positions = np.load(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_positions.npy'.format(epoch_len, step_size, n_trajs, im_availability))
        actions = np.load(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_actions.npy'.format(epoch_len, step_size, n_trajs, im_availability))
    except Exception as e:
        logging.critical('failed to load {}'.format(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_rooms.npy'.format(epoch_len, step_size, n_trajs, im_availability)))
        os.makedirs(base_path+'replay', exist_ok=True)
        logging.critical('Start working on the replay')

        rooms, positions, actions = env.static_replay(initial_actions)
        np.save(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_rooms.npy'.format(epoch_len, step_size, n_trajs, im_availability), rooms)
        np.save(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_positions.npy'.format(epoch_len, step_size, n_trajs, im_availability), positions)
        np.save(base_path + 'replay/initial_actions_blob_len_{}_step_{}_ntrajs__{}_im_avail_{}_actions.npy'.format(epoch_len, step_size, n_trajs, im_availability), actions)


    absolute_positions = (env.room_centers[rooms.astype(int)][:,:,:2] + positions)[:, 1:] # should be (n_trajs, epoch_len, 2)
    cumulated_actions = np.cumsum(actions, axis=1)

    if resetting_type == 'fixed':
        reset_every = int(1/im_availability)
        ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((batch_size, 1)).float()
    elif resetting_type == 'random':
        ims_to_perturb =  tch.bernoulli((1.-im_availability) * tch.ones(batch_size, epoch_len+1))

    ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

    # Now, for images to perturb, choose between "corruption" and "drop"
    # try:
    corrupt = tch.where(tch.bernoulli(corruption_rate * tch.ones(batch_size, epoch_len+1)).byte(), ims_to_perturb, tch.zeros(batch_size, epoch_len+1)).bool()
    drop = tch.logical_and(ims_to_perturb, tch.logical_not(corrupt))
    mask = (1.-drop.float()).unsqueeze(-1).repeat(1, 1, net.representation_size).float().to(net.device)
    # except:
    #     corrupt = tch.where(tch.bernoulli(corruption_rate * tch.ones(batch_size, epoch_len+1)).byte(), ims_to_perturb, tch.zeros(batch_size, epoch_len+1))
    #     drop = ims_to_perturb *(1.-corrupt)
    #     mask = (1.-drop.float()).unsqueeze(-1).repeat(1, 1, net.representation_size).float().to(net.device)

    time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[1]+1)
    cmap = plt.get_cmap('jet')
    colors = cmap(time_based_norm(range(epoch_len+1)))

    marker = '*'
    marker_true = '+'

    # This is for plot of representative neurons
    n_neurons = 5
    all_bkp_representations = np.zeros((n_seeds, n_trajs, epoch_len, n_neurons))
    all_internal_states = np.zeros((4, n_seeds, n_trajs, epoch_len, n_neurons))


    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            logging.critical('Starting seed {}'.format(start_seed+seed))
            env = FixedRewardWorld(**all_params['env_params'], seed=start_seed+seed)
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
            net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(start_seed+seed)), strict=False)

            errors_noiseless = np.zeros((n_trajs, epoch_len))
            errors_noiseless_no_images = np.zeros((n_trajs, epoch_len))

            errors_noisy = np.zeros((n_trajs, epoch_len))
            errors_noisy_no_images = np.zeros((n_trajs, epoch_len))

            n = all_params['net_params']['options']['representation_size']
            pearsons_visual = np.zeros((n_trajs, epoch_len, n))
            pearsons_noisy = np.zeros((n_trajs, epoch_len, n))
            pearsons_noisy_no_images = np.zeros((n_trajs, epoch_len, n))
            pearsons_noiseless = np.zeros((n_trajs, epoch_len, n))
            pearsons_noiseless_no_images = np.zeros((n_trajs, epoch_len, n))

            # First, do the noisy versions
            for batch_idx in range(n_trajs//batch_size):
                rooms_loc = deepcopy(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size])
                positions_loc = positions[batch_idx*batch_size:(batch_idx+1)*batch_size]

                images = env.get_images(rooms_loc, positions_loc)

                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                all_bkp_representations[seed, batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(representations[:, 1:, :n_neurons].detach().cpu().numpy()) # Only need to get this once
                pearsons_visual[batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(representations[:, 1:].detach().cpu().numpy()) # Only need to get this once
                actions_encodings =  net.get_z_encoding(tch.from_numpy(actions[batch_idx*batch_size:(batch_idx+1)*batch_size]
                                                        + noise*np.random.randn(*actions[batch_idx*batch_size:(batch_idx+1)*batch_size].shape)).view(batch_size * (epoch_len), 2).float().to(net.device))
                actions_encodings = actions_encodings.view(batch_size, (epoch_len), -1)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                outputs, _, internal_states = net.do_path_integration(representations, actions_encodings)
                outputs = outputs.detach().cpu().numpy()
                internal_states = internal_states.detach().cpu().numpy()
                all_internal_states[0, seed, batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states[:, :, :n_neurons])

                errors_noisy[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))
                pearsons_noisy[batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states) # First store the internal states in there, then only compute the pearsons
                del internal_states, outputs

                fully_perturbed_representations = deepcopy(representations)
                for t in tqdm(range(epoch_len)):
                    fully_perturbed_representations[:, t] = fully_perturbed_representations[:, t, tch.randperm(fully_perturbed_representations.shape[2])]
                fully_perturbed_representations[:, 0] = representations[:, 0]
                outputs_no_images, _, internal_states = net.do_path_integration(fully_perturbed_representations, actions_encodings)
                outputs_no_images = outputs_no_images.detach().cpu().numpy()
                internal_states = internal_states.detach().cpu().numpy()
                all_internal_states[1, seed, batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states[:, :, :n_neurons])

                errors_noisy_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs_no_images - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))
                pearsons_noisy_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states) # First store the internal states in there, then only compute the pearsons
                del internal_states, outputs_no_images

            # Then, the noiseless ones
            for batch_idx in range(n_trajs//batch_size):
                images = env.get_images(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size], positions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                actions_encodings =  net.get_z_encoding(tch.from_numpy(actions[batch_idx*batch_size:(batch_idx+1)*batch_size]).view(batch_size * (epoch_len), 2).float().to(net.device))
                actions_encodings = actions_encodings.view(batch_size, (epoch_len), -1)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                # logging.critical('786')

                outputs, g, internal_states = net.do_path_integration(representations, actions_encodings)
                # print(g[:,:, 0].min(), g[:,:, 0].max(), g[:,:, 0].mean())
                outputs = outputs.detach().cpu().numpy()
                internal_states = internal_states.detach().cpu().numpy()
                all_internal_states[2, seed, batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states[:, :, :n_neurons])
                errors_noiseless[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))
                pearsons_noiseless[batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states) # First store the internal states in there, then only compute the pearsons
                del internal_states, outputs

                fully_perturbed_representations = deepcopy(representations)
                for t in tqdm(range(epoch_len)):
                    fully_perturbed_representations[:, t] = fully_perturbed_representations[:, t, tch.randperm(fully_perturbed_representations.shape[2])]

                fully_perturbed_representations[:, 0] = representations[:, 0]
                outputs_no_images, g, internal_states = net.do_path_integration(fully_perturbed_representations, actions_encodings)

                outputs_no_images = outputs_no_images.detach().cpu().numpy()
                internal_states = internal_states.detach().cpu().numpy()
                all_internal_states[3, seed, batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states[:, :, :n_neurons])
                errors_noiseless_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs_no_images - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))
                pearsons_noiseless_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = deepcopy(internal_states) # First store the internal states in there, then only compute the pearsons
                del internal_states, outputs_no_images


            representations_tmp = deepcopy(pearsons_noiseless)
            # Now, actually compute the pearsons
            X = np.reshape(absolute_positions, (-1, 2))
            for id, container in zip(['pearsons_noisy', 'pearsons_noisy_no_images', 'pearsons_noiseless', 'pearsons_noiseless_no_images', 'pearsons_visual'],[pearsons_noisy, pearsons_noisy_no_images, pearsons_noiseless, pearsons_noiseless_no_images, pearsons_visual]):
                y = np.reshape(container, (-1, n))
                logging.critical(absolute_positions.shape)
                logging.critical(X.shape)
                logging.critical(container.shape)
                logging.critical(y.shape)
                model = LinearRegression()
                logging.critical('starting fit of linear regression')
                model.fit(X, y)
                preds = model.predict(X)
                scores = r2_score(y, preds, multioutput='raw_values')
                logging.critical('When computing, {} {}, min/max/mean/std: {}/{}/{}/{}'.format(id, scores.shape, scores.min(), scores.max(), scores.mean(), scores.std()))

                if id == 'pearsons_noisy':
                    pearsons_noisy = deepcopy(scores)
                elif id == 'pearsons_noisy_no_images':
                    pearsons_noisy_no_images = deepcopy(scores)
                elif id == 'pearsons_noiseless':
                    pearsons_noiseless = deepcopy(scores)
                elif id == 'pearsons_noiseless_no_images':
                    pearsons_noiseless_no_images = deepcopy(scores)
                elif id == 'pearsons_visual':
                    pearsons_visual = deepcopy(scores)


            os.makedirs(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}'.format(seed, epoch_len, step_size, n_trajs, im_availability), exist_ok=True)
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), errors_noiseless)
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), errors_noiseless_no_images)
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), errors_noisy)
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), errors_noisy_no_images)
            logging.critical('At save time, pearsons noisy shape {}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noisy.shape, pearsons_noisy.min(), pearsons_noisy.max(), pearsons_noisy.mean(), pearsons_noisy.std()))
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), pearsons_noisy)
            logging.critical('At save time, pearsons_noisy_no_images shape {}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noisy_no_images.shape, pearsons_noisy_no_images.min(), pearsons_noisy_no_images.max(), pearsons_noisy_no_images.mean(), pearsons_noisy_no_images.std()))
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), pearsons_noisy_no_images)
            logging.critical('At save time, pearsons_noiseless shape {}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noiseless.shape, pearsons_noiseless.min(), pearsons_noiseless.max(), pearsons_noiseless.mean(), pearsons_noiseless.std()))
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), pearsons_noiseless)
            logging.critical('At save time, pearsons_noiseless_no_images shape {}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noiseless_no_images.shape, pearsons_noiseless_no_images.min(), pearsons_noiseless_no_images.max(), pearsons_noiseless_no_images.mean(), pearsons_noiseless.std()))
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), pearsons_noiseless_no_images)

            logging.critical('At save time, pearsons_visual shape {}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_visual.shape, pearsons_visual.min(), pearsons_visual.max(), pearsons_visual.mean(), pearsons_visual.std()))
            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_visual.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), pearsons_visual)

            # Now in displacement coordinates
            X = np.reshape(cumulated_actions, (-1, 2))
            y = np.reshape(representations_tmp, (-1, n))
            model = LinearRegression()
            logging.critical('starting fit of linear regression')
            model.fit(X, y)
            preds = model.predict(X)
            scores = r2_score(y, preds, multioutput='raw_values')

            np.save(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_displacement_coordinates_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability), scores)







        seismic = plt.get_cmap('seismic')
        # plop = np.stack([absolute_positions]*4, axis=0)
        # logging.critical(plop.shape)
        all_global_positions = np.reshape(absolute_positions, (-1, 2))
        # Also do representation plots at the end
        # os.makedirs(path+'representation/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}'.format(epoch_len, step_size, n_trajs, im_availability), exist_ok=True)
        os.makedirs(figs_path+'representation/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}'.format(epoch_len, step_size, n_trajs, im_availability), exist_ok=True)
        for seed in range(n_seeds):
            for neuron in range(n_neurons):
                neuron_visual_representation = all_bkp_representations[ seed, :, :, neuron].flatten()
                norm = matplotlib.colors.Normalize(vmin=neuron_visual_representation.min(), vmax=neuron_visual_representation.max())

                fig, ax = plt.subplots(1, 1, figsize=(6,6))
                ax = env.render_template(ax_to_use=ax, add_goal=False)
                neuron_colors = seismic(norm(neuron_visual_representation)).reshape((-1, 4))
                ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))
                # fig.savefig(path+'representation/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}/seed_{}_neuron_{}.png'.format(epoch_len, step_size, n_trajs, im_availability, seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                fig.savefig(figs_path + 'representation/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}/seed_{}_neuron_{}.png'.format(epoch_len, step_size, n_trajs, im_availability, seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                plt.close('all')


        for idx, name in enumerate(['noisy', 'noisy_no_images', 'noiseless', 'noiseless_no_images']):
            # os.makedirs(path+'representation/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(epoch_len, step_size, n_trajs, im_availability)+name, exist_ok=True)
            os.makedirs(figs_path+'representation/{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(name, epoch_len, step_size, n_trajs, im_availability), exist_ok=True)
            for seed in range(n_seeds):
                for neuron in range(n_neurons):
                    neuron_activations = all_internal_states[idx, seed, :, :, neuron].flatten()

                    norm = matplotlib.colors.Normalize(vmin=neuron_activations.min(), vmax=neuron_activations.max())
                    fig, ax = plt.subplots(1, 1, figsize=(6,6))
                    ax = env.render_template(ax_to_use=ax, add_goal=False)
                    neuron_colors = seismic(norm(neuron_activations)).reshape((-1, 4))
                    ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))

                    # fig.savefig(path+'representation/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(epoch_len, step_size, n_trajs, im_availability)+name+'/seed_{}_neuron_{}.png'.format(seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                    fig.savefig((figs_path+'representation/{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(name, epoch_len, step_size, n_trajs, im_availability)+'/seed_{}_neuron_{}.png'.format(seed, neuron))) # need to save as png, otherwise it makes an absurdly large pdf
                    plt.close('all')




        # Now in displacement coordinates

        all_displacements = np.reshape(cumulated_actions, (-1, 2))
        os.makedirs(figs_path+'representation_in_displacement_coords/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(epoch_len, step_size, n_trajs, im_availability), exist_ok=True)

        for seed in range(n_seeds):
            for neuron in range(n_neurons):
                neuron_visual_representation = all_bkp_representations[seed, :, :, neuron].flatten()
                norm = matplotlib.colors.Normalize(vmin=neuron_visual_representation.min(), vmax=neuron_visual_representation.max())

                fig, ax = plt.subplots(1, 1, figsize=(6,6))
                # ax = env.render_template(ax_to_use=ax, add_goal=False)
                neuron_colors = seismic(norm(neuron_visual_representation)).reshape((-1, 4))
                # ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))
                ax.scatter(all_displacements[:,0], all_displacements[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))
                plt.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                # fig.savefig(path+'representation/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}/seed_{}_neuron_{}.png'.format(epoch_len, step_size, n_trajs, im_availability, seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                fig.savefig(figs_path+'representation_in_displacement_coords/visual/len_{}_step_{}_ntrajs__{}_im_avail_{}/seed_{}_neuron_{}.png'.format(epoch_len, step_size, n_trajs, im_availability, seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                plt.close('all')


        for idx, name in enumerate(['noisy', 'noisy_no_images', 'noiseless', 'noiseless_no_images']):
            # os.makedirs(path+'representation/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(epoch_len, step_size, n_trajs, im_availability)+name, exist_ok=True)
            os.makedirs(figs_path+'representation_in_displacement_coords/{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(name, epoch_len, step_size, n_trajs, im_availability), exist_ok=True)
            for seed in range(n_seeds):
                for neuron in range(n_neurons):
                    neuron_activations = all_internal_states[idx, seed, :, :, neuron].flatten()
                    norm = matplotlib.colors.Normalize(vmin=neuron_activations.min(), vmax=neuron_activations.max())
                    fig, ax = plt.subplots(1, 1, figsize=(6,6))
                    # ax = env.render_template(ax_to_use=ax, add_goal=False)
                    neuron_colors = seismic(norm(neuron_activations)).reshape((-1, 4))
                    plt.axis('off')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    # ax.scatter(all_global_positions[:,0], all_global_positions[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))
                    ax.scatter(all_displacements[:,0], all_displacements[:,1], c=neuron_colors, s=256000/(n_trajs*epoch_len))
                    fig.savefig(figs_path+'representation_in_displacement_coords/{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/'.format(name, epoch_len, step_size, n_trajs, im_availability)+'/seed_{}_neuron_{}.png'.format(seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                    plt.close('all')


            logging.critical(pearsons_noisy_no_images.shape)





def make_long_experiment_error_figures(map='SnakePath', layout='Default', exp_group='end_to_end', rep_size=512, protocol='default',
                        batch_size=64, n_trajs=512, epoch_len=100, step_size=.5, start_seed=0, n_seeds=4, im_availability=.1,
                        corruption_rate=.5, noise=0.05, resetting_type='fixed', use_reimplementation=None, collapsed=False,
                        reinference_errors = [0., .02, .04, .08], image_availabilities = [0., .2, .5, 1.]):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    base_path = 'out/{}_{}/'.format(map, layout)
    figs_path = base_path + 'long_experiment/figures/'
    os.makedirs(figs_path, exist_ok=True)

    training_type = '_'.join(protocol.split('/'))

    n_errors = len(reinference_errors)
    n_avails = len(image_availabilities)


    all_errors_noiseless = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
    all_errors_noiseless_no_images = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
    all_errors_noisy = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
    all_errors_noisy_no_images = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
    all_pearsons_noiseless = np.zeros((n_errors, n_avails,n_seeds, rep_size))
    all_pearsons_noiseless_no_images = np.zeros((n_errors, n_avails, n_seeds, rep_size))
    all_pearsons_noisy = np.zeros((n_errors, n_avails, n_seeds, rep_size))
    all_pearsons_noisy_no_images = np.zeros((n_errors, n_avails, n_seeds, rep_size))

    reset_every = int(1./im_availability)
    ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((n_trajs, 1)).float()
    ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

    for error_idx, train_reinference_error in enumerate(reinference_errors):
        for avail_idx, train_image_availability in enumerate(image_availabilities):
            path = 'out/{}_{}/{}/{}/error_{}_avail_{}/'.format(map, layout, exp_group, protocol, train_reinference_error, train_image_availability)
            with tch.set_grad_enabled(False):
                for seed in range(start_seed, start_seed+n_seeds):
                    # print(seed)
                    errors_noiseless = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    errors_noiseless_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    errors_noisy = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    errors_noisy_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    pearsons_noisy = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    pearsons_noisy_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy_no_images.npy'.format(seed,epoch_len, step_size, n_trajs, im_availability))
                    pearsons_noiseless = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    pearsons_noiseless_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                    pearsons_visual = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_visual.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))

                    seed = seed-start_seed
                    all_errors_noiseless[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless
                    all_errors_noiseless_no_images[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless_no_images
                    all_errors_noisy[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noisy
                    all_errors_noisy_no_images[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noisy_no_images
                    all_pearsons_noiseless[error_idx, avail_idx, seed] = pearsons_noiseless
                    all_pearsons_noiseless_no_images[error_idx, avail_idx, seed] = pearsons_noiseless_no_images
                    all_pearsons_noisy[error_idx, avail_idx, seed] = pearsons_noisy
                    all_pearsons_noisy_no_images[error_idx, avail_idx, seed] = pearsons_noisy_no_images

                    logging.critical('{}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noiseless.shape, pearsons_noiseless.min(), pearsons_noiseless.max(), pearsons_noiseless.mean(), pearsons_noiseless.std()))


            os.makedirs(figs_path+'im_availability_{}_noise_{}_resetting_{}/errors'.format(im_availability, noise, resetting_type), exist_ok=True)

            fig, axes = plt.subplots(1,2, figsize=(10,5))

            ax = axes[0]
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k', zorder=5, alpha=.5)
            plot_mean_std(ax, all_errors_noiseless_no_images[error_idx, avail_idx], label=r'Without images', c_line='tab:blue', c_fill='tab:blue')
            plot_mean_std(ax, all_errors_noiseless[error_idx, avail_idx], label=r'With images', c_line='tab:orange', c_fill='tab:orange')
            ax.set_xlabel('Time')
            ax.set_ylabel('PI error')
            ax.legend(loc=2, framealpha=1)
            ax.set_title(r'\centering Perfect reafference ($\epsilon=0$) First steps: ${:.2e}\pm{:.2e}$ Full length: ${:.2e}\pm{:.2e}$'.format(
                all_errors_noiseless[error_idx, avail_idx, :reset_every].mean(), all_errors_noiseless[error_idx, avail_idx, :reset_every].std(),
                all_errors_noiseless[error_idx, avail_idx].mean(), all_errors_noiseless[error_idx, avail_idx].std(),
            ))

            ax = axes[1]
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k', zorder=5, alpha=.5)
            plot_mean_std(ax, all_errors_noisy_no_images[error_idx, avail_idx], label=r'Without images', c_line='tab:blue', c_fill='tab:blue')
            plot_mean_std(ax, all_errors_noisy[error_idx, avail_idx], label=r'With images', c_line='tab:orange', c_fill='tab:orange')
            ax.set_xlabel('Time')
            ax.set_ylabel('PI error')
            ax.legend(loc=2, framealpha=1)
            fig.tight_layout()
            fig.savefig(figs_path+'im_availability_{}_noise_{}_resetting_{}/errors/{}_err_{}_avail_{}.pdf'.format(im_availability, noise, resetting_type, training_type, train_reinference_error, train_image_availability))
            plt.close(fig)


            os.makedirs(figs_path+'im_availability_{}_noise_{}_resetting_{}/pearsons'.format(im_availability, noise, resetting_type), exist_ok=True)
            fig, axes = plt.subplots(1,2, figsize=(10,3.5))

            ax = axes[0]
            ax.hist(all_pearsons_noiseless[error_idx, avail_idx].flatten(), range=[0,1], bins=100)
            ax.set_xlabel('Normalized reconstruction error')
            ax.set_ylabel('Bin count')
            ax.set_title(r' \centering Perfect reafference ($\epsilon=0$) \\ $R^2$ is ${:.2} \pm {:.2}$\\'.format(all_pearsons_noiseless[error_idx, avail_idx].flatten().mean(), all_pearsons_noiseless[error_idx, avail_idx].flatten().std()))

            ax = axes[1]
            ax.hist(all_pearsons_noisy[error_idx, avail_idx].flatten(), range=[0,1], bins=100)
            ax.set_xlabel('Normalized reconstruction error')
            ax.set_ylabel('Bin count')
            ax.set_title(r'\centering Noisy reafference ($\epsilon={:.3}$)  \\ $R^2$ is ${:.2} \pm {:.2}$\\'.format(noise, all_pearsons_noisy[error_idx, avail_idx].flatten().mean(), all_pearsons_noisy[error_idx, avail_idx].flatten().std()))

            fig.savefig(figs_path+'im_availability_{}_noise_{}_resetting_{}/pearsons/{}_err_{}_avail_{}.pdf'.format(im_availability, noise, resetting_type, training_type, train_reinference_error, train_image_availability))
            plt.close(fig)





def do_latex_table(protocols=['offshelf_LSTM/pretrained', 'minimal_model/all_losses', 'minimal_model/no_fb_losses', 'offshelf_LSTM/pretrained_no_start_rep', 'offshelf_LSTM/use_start_rep_no_pretrain']
                    , map='SnakePath', layout='Default', exp_group='end_to_end', rep_size=512, protocol='default', batch_size=64, n_trajs=512, epoch_len=100, table_name='default',
                     step_size=.5, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', use_reimplementation=None, collapsed=False, reinference_errors = [0., .02, .04, .08], image_availabilities = [0., .2, .5, 1.]):
    dict = {
        'protocol': [],

        'train availability': [],
        'test availability': [],
        'train noise': [],
        'test noise': [],

        'errors_noiseless_short_mean': [],
        'errors_noisy_short_mean': [],
        'errors_noiseless_no_images_short_mean': [],
        'errors_noisy_no_images_short_mean': [],
        'errors_noiseless_short_std': [],
        'errors_noisy_short_std': [],
        'errors_noiseless_no_images_short_std': [],
        'errors_noisy_no_images_short_std': [],

        'errors_noiseless_long_mean': [],
        'errors_noisy_long_mean': [],
        'errors_noiseless_no_images_long_mean': [],
        'errors_noisy_no_images_long_mean': [],
        'errors_noiseless_long_std': [],
        'errors_noisy_long_std': [],
        'errors_noiseless_no_images_long_std': [],
        'errors_noisy_no_images_long_std': [],

        'errors_ratio_noiseless': [],

        'pearsons_noiseless_mean': [],
        'pearsons_noisy_mean': [],
        'pearsons_noiseless_no_images_mean': [],
        'pearsons_noisy_no_images_mean': [],
        'pearsons_noiseless_std': [],
        'pearsons_noisy_std': [],
        'pearsons_noiseless_no_images_std': [],
        'pearsons_noisy_no_images_std': [],

        'pearsons_visual_mean': [],
        'pearsons_visual_std': [],

    }

    n_errors = len(reinference_errors)
    n_avails = len(image_availabilities)

    blob_errors_noiseless = np.zeros((len(protocols), n_errors, n_avails, n_trajs*n_seeds, epoch_len))
    blob_errors_noiseless = np.zeros((len(protocols), n_errors, n_avails, n_seeds, epoch_len))
    blob_pearsons_noiseless = np.zeros((len(protocols), n_errors, n_avails,n_seeds, rep_size))
    blob_pearsons_visual = np.zeros((len(protocols), n_errors, n_avails, n_seeds, rep_size))

    base_path = 'out/{}_{}/'.format(map, layout)
    for protocol_idx, protocol in enumerate(protocols):
        training_type = '_'.join(protocol.split('/'))

        all_errors_noiseless = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
        all_errors_noiseless_no_images = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
        all_errors_noisy = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
        all_errors_noisy_no_images = np.zeros((n_errors, n_avails, n_trajs*n_seeds, epoch_len))
        all_pearsons_noiseless = np.zeros((n_errors, n_avails,n_seeds, rep_size))
        all_pearsons_noiseless_no_images = np.zeros((n_errors, n_avails, n_seeds, rep_size))
        all_pearsons_noisy = np.zeros((n_errors, n_avails, n_seeds, rep_size))
        all_pearsons_noisy_no_images = np.zeros((n_errors, n_avails, n_seeds, rep_size))
        all_pearsons_visual = np.zeros((n_errors, n_avails, n_seeds, rep_size))

        reset_every = int(1./im_availability)
        ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((n_trajs, 1)).float()
        ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

        for error_idx, train_reinference_error in enumerate(reinference_errors):
            for avail_idx, train_image_availability in enumerate(image_availabilities):
                path = 'out/{}_{}/{}/{}/error_{}_avail_{}/'.format(map, layout, exp_group, protocol, train_reinference_error, train_image_availability)
                with tch.set_grad_enabled(False):
                    for seed in range(n_seeds):
                        errors_noiseless = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        errors_noiseless_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        errors_noisy = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        errors_noisy_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/errors_noisy_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        pearsons_noisy = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        pearsons_noisy_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noisy_no_images.npy'.format(seed,epoch_len, step_size, n_trajs, im_availability))
                        pearsons_noiseless = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        pearsons_noiseless_no_images = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_noiseless_no_images.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        pearsons_visual = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_visual.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))

                        all_errors_noiseless[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless
                        all_errors_noiseless_no_images[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless_no_images
                        all_errors_noisy[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noisy
                        all_errors_noisy_no_images[error_idx, avail_idx, seed*n_trajs:(seed+1)*n_trajs] = errors_noisy_no_images
                        all_pearsons_noiseless[error_idx, avail_idx, seed] = pearsons_noiseless
                        all_pearsons_noiseless_no_images[error_idx, avail_idx, seed] = pearsons_noiseless_no_images
                        all_pearsons_noisy[error_idx, avail_idx, seed] = pearsons_noisy
                        all_pearsons_noisy_no_images[error_idx, avail_idx, seed] = pearsons_noisy_no_images
                        all_pearsons_visual[error_idx, avail_idx, seed] = pearsons_visual


                        logging.critical('{}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noiseless.shape, pearsons_noiseless.min(), pearsons_noiseless.max(), pearsons_noiseless.mean(), pearsons_noiseless.std()))


                dict['protocol'].append(protocol)
                dict['train availability'].append(train_image_availability)
                dict['test availability'].append(im_availability)
                dict['train noise'].append(train_reinference_error)
                dict['test noise'].append(noise)


                errors_short_noiseless = all_errors_noiseless[error_idx, avail_idx, :, :reset_every].flatten()
                errors_short_noisy = all_errors_noisy[error_idx, avail_idx, :, :reset_every].flatten()
                errors_short_noiseless_no_images = all_errors_noiseless_no_images[error_idx, avail_idx, :, :reset_every].flatten()
                errors_short_noisy_no_images = all_errors_noisy_no_images[error_idx, avail_idx, :, :reset_every].flatten()
                dict['errors_noiseless_short_mean'].append(errors_short_noiseless.mean())
                dict['errors_noisy_short_mean'].append(errors_short_noisy.mean())
                dict['errors_noiseless_no_images_short_mean'].append(errors_short_noiseless_no_images.mean())
                dict['errors_noisy_no_images_short_mean'].append(errors_short_noisy_no_images.mean())
                dict['errors_noiseless_short_std'].append(errors_short_noiseless.std())
                dict['errors_noisy_short_std'].append(errors_short_noisy.std())
                dict['errors_noiseless_no_images_short_std'].append(errors_short_noiseless_no_images.std())
                dict['errors_noisy_no_images_short_std'].append(errors_short_noisy_no_images.std())

                errors_long_noiseless = all_errors_noiseless[error_idx, avail_idx].flatten()
                errors_long_noisy = all_errors_noisy[error_idx, avail_idx].flatten()
                errors_long_noiseless_no_images = all_errors_noiseless_no_images[error_idx, avail_idx].flatten()
                errors_long_noisy_no_images = all_errors_noisy_no_images[error_idx, avail_idx].flatten()
                dict['errors_noiseless_long_mean'].append(errors_long_noiseless.mean())
                dict['errors_noisy_long_mean'].append(errors_long_noisy.mean())
                dict['errors_noiseless_no_images_long_mean'].append(errors_long_noiseless_no_images.mean())
                dict['errors_noisy_no_images_long_mean'].append(errors_long_noisy_no_images.mean())
                dict['errors_noiseless_long_std'].append(errors_long_noiseless.std())
                dict['errors_noisy_long_std'].append(errors_long_noisy.std())
                dict['errors_noiseless_no_images_long_std'].append(errors_long_noiseless_no_images.std())
                dict['errors_noisy_no_images_long_std'].append(errors_long_noisy_no_images.std())


                dict['errors_ratio_noiseless'].append(errors_long_noiseless.mean()/errors_short_noiseless.mean())

                pearsons_noiseless = all_pearsons_noiseless[error_idx, avail_idx]
                pearsons_noiseless_no_images = all_pearsons_noiseless_no_images[error_idx, avail_idx]
                pearsons_noisy = all_pearsons_noisy[error_idx, avail_idx]
                pearsons_noisy_no_images = all_pearsons_noisy_no_images[error_idx, avail_idx]
                pearsons_visual = all_pearsons_visual[error_idx, avail_idx]

                dict['pearsons_noiseless_mean'].append(pearsons_noiseless.mean())
                dict['pearsons_noiseless_std'].append(pearsons_noiseless.std())
                dict['pearsons_noisy_mean'].append(pearsons_noisy.mean())
                dict['pearsons_noisy_std'].append(pearsons_noisy.std())
                dict['pearsons_noiseless_no_images_mean'].append(pearsons_noiseless_no_images.mean())
                dict['pearsons_noiseless_no_images_std'].append(pearsons_noiseless_no_images.std())
                dict['pearsons_noisy_no_images_mean'].append(pearsons_noisy_no_images.mean())
                dict['pearsons_noisy_no_images_std'].append(pearsons_noisy_no_images.std())
                dict['pearsons_visual_mean'].append(pearsons_visual.mean())
                dict['pearsons_visual_std'].append(pearsons_visual.std())

                blob_errors_noiseless[protocol_idx] = all_errors_noiseless.mean(axis=2)
                blob_pearsons_noiseless[protocol_idx] = all_pearsons_noiseless
                blob_pearsons_visual[protocol_idx] = all_pearsons_visual

    for key, val in dict.items():
        print(key, len(val))

    frame = pd.DataFrame(dict)
    print(frame)
    frame.to_csv(base_path+exp_group+'/data_im_availability_{}_noise_{}_resetting_{}.csv'.format(im_availability, noise, resetting_type, training_type))


    print(base_path+exp_group+'/{}_table.tex'.format(table_name))
    with open(base_path+exp_group+'/{}_table.tex'.format(table_name), 'w+') as f:
        f.write(' ')
        for protocol in protocols:
            f.write(' & ' +protocol)
        f.write('\\\\\n')

        f.write('Error (short)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['errors_noiseless_short_mean'][idx], frame['errors_noiseless_short_std'][idx]))
        f.write('\\\\\n')

        f.write('Error (long)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['errors_noiseless_long_mean'][idx], frame['errors_noiseless_long_std'][idx]))
        f.write('\\\\\n')


        f.write('$R^2$ (visual)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['pearsons_visual_mean'][idx], frame['pearsons_visual_std'][idx]))
        f.write('\\\\\n')

        f.write('$R^2$ (PI)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['pearsons_noiseless_mean'][idx], frame['pearsons_noiseless_std'][idx]))
        f.write('\\\\\n')

    from scipy.stats import ttest_ind
    from seaborn import heatmap


    quantity_labels = ['errors_short', 'errors_long', 'pearsons_visual', 'pearsons_pi']
    protocol_labels = [' '.join(p.split('_')) for p in protocols]


    for data, label in zip([blob_errors_noiseless[:,-1, -1, :, :reset_every],
        blob_errors_noiseless, blob_pearsons_visual, blob_pearsons_noiseless],
        quantity_labels):

        os.makedirs(base_path+exp_group+'/stats', exist_ok=True)
        test_results = np.zeros((len(protocols), len(protocols)))
        for i in range(len(protocols)):
            for j in range(i, len(protocols)):
                protocol_one, protocol_two = protocols[i], protocols[j]
                # print(ttest_ind(data[i].flatten(), data[j].flatten(), equal_var=False).pvalue)
                test_results[i,j] = ttest_ind(data[i].flatten(), data[j].flatten(), equal_var=False).pvalue
                test_results[j,i] = test_results[i,j]

                fig, axes = plt.subplots(1,2, figsize=(10,5))
                axes[0].hist(data[i].flatten())
                axes[1].hist(data[j].flatten())

                plt.savefig(base_path+exp_group+'/stats/{}_between_{}_and_{}.pdf'.format(label, '_'.join(protocols[i].split('/')), '_'.join(protocols[j].split('/'))))

        plt.figure()
        ax = heatmap(test_results,annot=True, xticklabels=protocol_labels, yticklabels=protocol_labels)#, fmt='{:.2}')
        plt.savefig((base_path+exp_group+'/stats/{}_{}_welch_test_results.pdf'.format(table_name, label)))


def do_coordinate_systems_table(protocols=['offshelf_LSTM/pretrained', 'minimal_model/all_losses', 'minimal_model/no_fb_losses', 'offshelf_LSTM/pretrained_no_start_rep', 'offshelf_LSTM/use_start_rep_no_pretrain']
                    , map='SnakePath', layout='Default', exp_group='end_to_end', rep_size=512, protocol='default', batch_size=64, n_trajs=512, epoch_len=100, table_name='default',
                     step_size=.5, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', use_reimplementation=None, collapsed=False,
                     reinference_errors = [0., .02, .04, .08], image_availabilities = [0., .2, .5, 1.]):
    dict = {
        'protocol': [],

        'pearsons_noiseless_mean': [],
        'pearsons_noiseless_std': [],
        'protocol': [],
        'train availability': [],
        'test availability': [],
        'train noise': [],
        'test noise': [],

    }

    n_errors = len(reinference_errors)
    n_avails = len(image_availabilities)

    blob_pearsons_noiseless = np.zeros((len(protocols), n_errors, n_avails,n_seeds, rep_size))


    base_path = 'out/{}_{}/'.format(map, layout)
    for protocol_idx, protocol in enumerate(protocols):
        training_type = '_'.join(protocol.split('/'))

        all_pearsons_noiseless = np.zeros((n_errors, n_avails,n_seeds, rep_size))

        reset_every = int(1./im_availability)
        ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((n_trajs, 1)).float()
        ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

        for error_idx, train_reinference_error in enumerate(reinference_errors):
            for avail_idx, train_image_availability in enumerate(image_availabilities):
                path = 'out/{}_{}/{}/{}/error_{}_avail_{}/'.format(map, layout, exp_group, protocol, train_reinference_error, train_image_availability)
                with tch.set_grad_enabled(False):
                    for seed in range(n_seeds):
                        pearsons_noiseless = np.load(path+'seed{}/len_{}_step_{}_ntrajs__{}_im_avail_{}/pearsons_displacement_coordinates_noiseless.npy'.format(seed, epoch_len, step_size, n_trajs, im_availability))
                        all_pearsons_noiseless[error_idx, avail_idx, seed] = pearsons_noiseless
                        logging.critical('{}, min/max/mean/std: {}/{}/{}/{}'.format(pearsons_noiseless.shape, pearsons_noiseless.min(), pearsons_noiseless.max(), pearsons_noiseless.mean(), pearsons_noiseless.std()))

                dict['protocol'].append(protocol)
                dict['train availability'].append(train_image_availability)
                dict['test availability'].append(im_availability)
                dict['train noise'].append(train_reinference_error)
                dict['test noise'].append(noise)
                pearsons_noiseless = all_pearsons_noiseless[error_idx, avail_idx]
                dict['pearsons_noiseless_mean'].append(pearsons_noiseless.mean())
                dict['pearsons_noiseless_std'].append(pearsons_noiseless.std())
                blob_pearsons_noiseless[protocol_idx] = all_pearsons_noiseless


    for key, val in dict.items():
        print(key, len(val))

    frame = pd.DataFrame(dict)
    print(frame)
    frame.to_csv(base_path+exp_group+'/data_im_availability_{}_noise_{}_resetting_{}.csv'.format(im_availability, noise, resetting_type, training_type))


    print(base_path+exp_group+'/{}_coordinate_systems_table.tex'.format(table_name))
    with open(base_path+exp_group+'/{}_coordinate_systems_table.tex'.format(table_name), 'w+') as f:
        f.write(' ')
        for protocol in protocols:
            f.write(' & ' +protocol)
        f.write('\\\\\n')

        f.write('$R^2$ (absolute)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['pearsons_noiseless_mean'][idx], frame['pearsons_noiseless_std'][idx]))
        f.write('\\\\\n')

        f.write('$R^2$ (displacement)')
        for idx, protocol in enumerate(protocols): # works because we know the order...
            # print(frame['pearsons_noiseless_mean'])
            f.write(' & {:.2} $\pm$ {:.2}'.format(frame['pearsons_noiseless_mean'][idx], frame['pearsons_noiseless_std'][idx]))
        f.write('\\\\\n')



def make_with_out_forward_figure(map='SnakePath', layout='Default', exp_group='end_to_end', rep_size=512, n_reps=5, protocols=['without_forward', 'with_forward'],
                        batch_size=512, n_points=4096, epoch_len=100, step_size=.5, start_seed=0, n_seeds=8):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    seismic = plt.get_cmap('seismic')

    base_path = 'out/{}_{}/'.format(map, layout)

    all_representations = np.zeros((2, 3*n_points, n_seeds, rep_size)) # Don't waste any of our computations
    all_global_positions = np.zeros((3*n_points, n_seeds, 2))

    displacements = np.zeros((2, n_points, n_seeds, 2))
    displacements_generalization = np.zeros((2, n_points, n_seeds, 2))

    errors = np.zeros((2, n_points, n_seeds))
    errors_generalization = np.zeros((2, n_points, n_seeds))

    all_scores = np.zeros((2, n_seeds, rep_size))

    semi_base_path = 'out/{}_{}/{}/'.format(map, layout, exp_group)
    for prot_idx, protocol in enumerate(protocols):
        path = 'out/{}_{}/{}/{}/'.format(map, layout, exp_group, protocol)
        figs_path = path + 'figures/'
        errors_figs_path = path + 'figures/errors/'
        reps_figs_path = path + 'figures/reps/'
        os.makedirs(figs_path, exist_ok=True)
        os.makedirs(errors_figs_path, exist_ok=True)
        os.makedirs(reps_figs_path, exist_ok=True)



        print(path)
        with open(path+'full_params.json') as f:
            all_params = json.load(f)
            env = FixedRewardWorld(**all_params['env_params'], seed=777)
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])

        start_positions = np.random.uniform(-env.scale, env.scale, size=(n_points, 2))
        start_rooms = np.random.randint(env.n_rooms, size=(n_points,))

        # This will test in training conditions
        actions = np.random.uniform(-step_size, step_size, size=(n_points, 1, 2))
        rooms, positions, actions = env.static_replay(actions, start_rooms=start_rooms, start_pos=start_positions)
        # new_positions = positions[:, 1]
        new_rooms, new_positions = rooms[:, 1], positions[:, 1]

        # For generalization, take any "second" position in the environment
        new_positions_generalization = np.random.uniform(-env.scale, env.scale, size=(n_points, 2))
        new_rooms_generalization = np.random.randint(env.n_rooms, size=(n_points,))

        # start_positions = np.random.uniform(-env.scale, env.scale, size=(n_points, 2))
        # start_rooms = np.random.randint(env.n_rooms, size=(n_points,))
        #
        # # This will test in training conditions
        # actions = np.random.uniform(-step_size, step_size, size=(n_points, 1, 2))
        # rooms, positions, actions = env.static_replay(actions, start_rooms=start_rooms, start_pos=start_positions)
        # # new_positions = positions[:, 1]
        # new_rooms, new_positions = rooms[:, 1], positions[:, 1]
        #
        # # For generalization, take any "second" position in the environment
        # new_positions_generalization = np.random.uniform(-env.scale, env.scale, size=(n_points, 2))
        # new_rooms_generalization = np.random.randint(env.n_rooms, size=(n_points,))
        #
        #
        #
        #
        # all_representations = np.zeros((3*n_points, n_seeds, rep_size)) # Don't waste any of our computations
        # all_global_positions = np.zeros((3*n_points, n_seeds, 2))
        #
        # displacements = np.zeros((n_points, n_seeds, 2))
        # displacements_generalization = np.zeros((n_points, n_seeds, 2))
        #
        # errors = np.zeros((n_points, n_seeds))
        # errors_generalization = np.zeros((n_points, n_seeds))
        #
        # all_scores = np.zeros((n_seeds, rep_size))

        for seed in range(start_seed, start_seed+n_seeds):
            with open(path+'full_params.json') as f:
                all_params = json.load(f)
                env = FixedRewardWorld(**all_params['env_params'], seed=777)
                net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])

            # print(semi_base_path+'seed{}/best_net.tch'.format(seed))
            # net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(seed)))
            net.load_state_dict(tch.load(path+'seed{}/final_net.tch'.format(seed)))

            start_images = env.get_images(start_rooms, start_positions)
            start_reps = net.get_representation(start_images)
            all_representations[prot_idx, :n_points, seed] = start_reps.detach().cpu().numpy()
            all_global_positions[:n_points, seed] = env.room_centers[start_rooms] + start_positions

            new_images = env.get_images(new_rooms, new_positions)
            new_reps = net.get_representation(new_images)
            all_representations[prot_idx, n_points:2*n_points, seed] = new_reps.detach().cpu().numpy()
            all_global_positions[n_points:2*n_points, seed] = env.room_centers[new_rooms.astype(int)] + new_positions


            new_images_generalization = env.get_images(new_rooms_generalization, new_positions_generalization)
            new_reps_generalization = net.get_representation(new_images_generalization)
            all_representations[prot_idx, 2*n_points:3*n_points, seed] = new_reps_generalization.detach().cpu().numpy()
            all_global_positions[2*n_points:3*n_points, seed] = env.room_centers[new_rooms_generalization] + new_positions_generalization

            for neuron in range(n_reps):
                neuron_visual_representation = all_representations[prot_idx, :, seed, neuron].flatten()
                norm = matplotlib.colors.Normalize(vmin=neuron_visual_representation.min(), vmax=neuron_visual_representation.max())

                fig, ax = plt.subplots(1, 1, figsize=(6,6))
                ax = env.render_template(ax_to_use=ax)
                neuron_colors = seismic(norm(neuron_visual_representation)).reshape((-1, 4))
                # ax.scatter(all_global_positions[:,seed,0], all_global_positions[:,seed,1], c=neuron_colors, s=64000/(n_points**2))
                ax.scatter(all_global_positions[:,seed,0], all_global_positions[:,seed,1], c=neuron_colors, s=64000/(n_points))
                fig.savefig(reps_figs_path+'seed_{}_neuron_{}.png'.format(seed, neuron)) # need to save as png, otherwise it makes an absurdly large pdf
                plt.close('all')

                X = np.reshape(all_global_positions[:, seed], (-1, 2))
                y = np.reshape(all_representations[prot_idx, :, seed], (-1, rep_size))
                model = LinearRegression()
                logging.critical('starting fit of linear regression')
                model.fit(X, y)
                preds = model.predict(X)
                all_scores[prot_idx, seed] = r2_score(y, preds, multioutput='raw_values')


            displacements[prot_idx, :, seed] = net.backward_model(start_reps, new_reps).detach().cpu().numpy()
            displacements_generalization[prot_idx, :, seed] = net.backward_model(start_reps, new_reps_generalization).detach().cpu().numpy()

            errors[prot_idx, :, seed] = np.sqrt(((displacements[prot_idx, :, seed]-actions[:,0])**2).sum(axis=-1))
            errors_generalization[prot_idx, :, seed] = np.sqrt(((displacements[prot_idx, :, seed]-(all_global_positions[2*n_points:3*n_points, seed]-all_global_positions[:n_points, seed]))**2).sum(axis=-1))

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].hist(errors[prot_idx, :, seed].flatten(), bins=100)
            axes[0].set_xlabel('Error (training conditions)')
            axes[1].hist(errors_generalization[prot_idx, :, seed].flatten(), bins=100)
            axes[1].set_xlabel('Error (generalization conditions)')
            plt.savefig(errors_figs_path+'seed_{}.pdf'.format(seed))

        print('error figs path: ', errors_figs_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].hist(errors[prot_idx].flatten(), bins=100)
        axes[0].set_xlabel('Error (training conditions)')
        axes[1].hist(errors_generalization[prot_idx].flatten(), bins=100)
        axes[1].set_xlabel('Error (generalization conditions)')
        plt.savefig(errors_figs_path+'general.pdf')


        plt.figure()
        plt.hist(all_scores[prot_idx].flatten(), bins=100, range=(0,1))
        plt.savefig(errors_figs_path+'correlations.png')

    # Comparison plots and table
    # print(path'/table.tex')
    with open(semi_base_path+'/table.tex', 'w+') as f:
        f.write(' & Error (training) & ')
        f.write('Error (generalization) & ')
        f.write('$R^2$ (visual) ')
        f.write('\\\\\n')

        f.write('Without direct')
        f.write(' & {:.2} $\pm$ {:.2}'.format(errors[0].mean(), errors[0].std()))
        f.write(' & {:.2} $\pm$ {:.2}'.format(errors_generalization[0].mean(), errors_generalization[0].std()))
        f.write(' & {:.2} $\pm$ {:.2}'.format(all_scores[0].mean(), all_scores[0].std()))
        f.write('\\\\\n')

        f.write('With direct')
        f.write(' & {:.2} $\pm$ {:.2}'.format(errors[1].mean(), errors[1].std()))
        f.write(' & {:.2} $\pm$ {:.2}'.format(errors_generalization[1].mean(), errors_generalization[1].std()))
        f.write(' & {:.2} $\pm$ {:.2}'.format(all_scores[1].mean(), all_scores[1].std()))
        f.write('\\\\\n')


if __name__ == '__main__':
    ########################################################
    ########################################################
    ################# Representation study #################
    ########################################################
    ########################################################

    # exp_group = 'with_out_direct_model'
    # protocols = [
    #             'with_forward',
    #             'without_forward',
    #             ]
    #
    #
    # make_with_out_forward_figure(map='SnakePath', layout='Default', exp_group=exp_group, rep_size=512, protocols=protocols, start_seed=0, n_seeds=8)


    ########################################################
    ########################################################
    ################## SnakePath Default ###################
    ########################################################
    ########################################################

    reinference_errors = [0.]
    image_availabilities = [.2]
    test_avail = .2

    # protocols = [
    #         'offshelf_LSTM/pretrained/',              # For LSTM comparison
    #         'offshelf_LSTM/pretrained_no_start_rep/',
    #         'offshelf_LSTM/use_start_rep_no_pretrain/',
    #         'hybrid_LSTM/scratch_high_fb/',
    #
    #         'minimal_model/all_losses/', # For main results
    #         'minimal_model/no_fb_losses/',
    #         'offshelf_LSTM/vanilla/',
    #         'hybrid_LSTM/pretrained/',
    #             ]


    protocols = [
    #             # # Main table
    #             'minimal_model/all_losses/',
    #             'minimal_model/no_fb_losses/',
    #             'offshelf_LSTM/vanilla/',
    #             'hybrid_LSTM/pretrained/',
    #
    #             # # Additional table
                'offshelf_LSTM/pretrained_no_start_rep/',
                'offshelf_LSTM/use_start_rep_no_pretrain/',
                'offshelf_LSTM/pretrained/',
                'hybrid_LSTM/scratch_high_fb/',
                ]



    for protocol in protocols:
        for reinference_error in reinference_errors:
            for image_availability in image_availabilities:
                # offline_study_errors(map='SnakePath', layout='Default', exp_group='long_experiment', protocol=protocol+'error_{}_avail_{}'.format(reinference_error, image_availability), epoch_len=100, step_size=.5, batch_size=16, n_trajs=512, start_seed=0, n_seeds=4, im_availability=test_avail, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
                offline_study_errors(map='SnakePath', layout='Default', exp_group='long_experiment', protocol=protocol+'error_{}_avail_{}'.format(reinference_error, image_availability), epoch_len=100, step_size=.5, batch_size=16, n_trajs=512, start_seed=0, n_seeds=8, im_availability=test_avail, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)


    for protocol in protocols:
       # make_long_experiment_error_figures(map='SnakePath', layout='Default', exp_group='long_experiment', protocol=protocol, collapsed=True, im_availability=.2, reinference_errors=reinference_errors, image_availabilities=image_availabilities, start_seed=0, n_seeds=4)
       make_long_experiment_error_figures(map='SnakePath', layout='Default', exp_group='long_experiment', protocol=protocol, collapsed=True, im_availability=.2, reinference_errors=reinference_errors, image_availabilities=image_availabilities, start_seed=0, n_seeds=8)


    # protocols = [
    #             'minimal_model/all_losses/',
    #             'minimal_model/no_fb_losses/',
    #             'offshelf_LSTM/vanilla/',
    #             'hybrid_LSTM/pretrained/',
    #             ]
    # table_name = 'minimal_model'
    #
    # do_latex_table(map='SnakePath', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)
    # do_coordinate_systems_table(map='SnakePath', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)

    protocols = [
                'offshelf_LSTM/pretrained_no_start_rep/',
                'offshelf_LSTM/use_start_rep_no_pretrain/',
                'offshelf_LSTM/pretrained/',
                'hybrid_LSTM/scratch_high_fb/', # Add "end-to-end"
                ]
    table_name = 'lstm_details'

    do_latex_table(map='SnakePath', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)
    do_coordinate_systems_table(map='SnakePath', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)


    ########################################################
    ########################################################
    ################# DoubleDonut Default ##################
    ########################################################
    ########################################################



    # reinference_errors = [0.]
    # image_availabilities = [.2]
    # test_avail = .2
    #
    #
    # protocols = [
    #             'minimal_model/all_losses/',
    #             'minimal_model/no_fb_losses/',
    #             'offshelf_LSTM/vanilla/',
    #             'hybrid_LSTM/pretrained_high_fb/',
    # ]
    # #
    # for protocol in protocols:
    #     for reinference_error in reinference_errors:
    #         for image_availability in image_availabilities:
    #             # offline_study_errors(map='DoubleDonut', layout='Default', exp_group='long_experiment', protocol=protocol+'error_{}_avail_{}'.format(reinference_error, image_availability), epoch_len=100, step_size=.5, batch_size=16, n_trajs=512, start_seed=0, n_seeds=4, im_availability=test_avail, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
    #             offline_study_errors(map='DoubleDonut', layout='Default', exp_group='long_experiment', protocol=protocol+'error_{}_avail_{}'.format(reinference_error, image_availability), epoch_len=100, step_size=.5, batch_size=16, n_trajs=512, start_seed=0, n_seeds=8, im_availability=test_avail, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
    #
    # for protocol in protocols:
    #    # make_long_experiment_error_figures(map='DoubleDonut', layout='Default', exp_group='long_experiment', protocol=protocol, collapsed=True, im_availability=.2, reinference_errors=reinference_errors, image_availabilities=image_availabilities, start_seed=0, n_seeds=4)
    #    make_long_experiment_error_figures(map='DoubleDonut', layout='Default', exp_group='long_experiment', protocol=protocol, collapsed=True, im_availability=.2, reinference_errors=reinference_errors, image_availabilities=image_availabilities, start_seed=0, n_seeds=8)
    #
    #
    # protocols = [
    #             'minimal_model/all_losses/',
    #             'minimal_model/no_fb_losses/',
    #             'offshelf_LSTM/vanilla/',
    #             'hybrid_LSTM/pretrained_high_fb/',
    #             ]
    # table_name = 'minimal_model'
    #
    # do_latex_table(map='DoubleDonut', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)
    # do_coordinate_systems_table(map='DoubleDonut', n_seeds=8, reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)



    # protocols = [
    #             'hybrid_LSTM/scratch_high_fb/',
    #             'offshelf_LSTM/use_start_rep_no_pretrain/',
    #             'offshelf_LSTM/pretrained/',
    #             'offshelf_LSTM/pretrained_no_start_rep/',
    #
    #             # Add "end-to-end"
    #             ]
    # table_name = 'lstm_details'
    #
    # do_latex_table(map='DoubleDonut', reinference_errors=[0.], image_availabilities=[.2], layout='Default', exp_group='long_experiment', im_availability=.2, protocols=protocols, table_name=table_name)
