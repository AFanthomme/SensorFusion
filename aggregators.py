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


def offline_study_errors(map='SnakePath', layout='Default', exp_group='end_to_end', protocol='default', batch_size=64, n_trajs=512, epoch_len=50, step_size=.5, start_seed=0, n_seeds=4, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', use_reimplementation=None, collapsed=False):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    path = 'out/{}_{}/{}/{}/'.format(map, layout, exp_group, protocol)

    with open(path+'full_params.json') as f:
        all_params = json.load(f)
        env = FixedRewardWorld(**all_params['env_params'], seed=777)
        # env_legacy = LegacyWorld(**all_params['env_params'], seed=777)
        # if use_reimplementation is None:
        net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
        # elif use_reimplementation == 'Small':
        #     net = ReimplementationPathIntegrator(**all_params['net_params']['options'])
        # elif use_reimplementation == 'Big':
        #     net = BigReimplementationPathIntegrator(**all_params['net_params']['options'])

    initial_actions = step_size * np.random.randn(n_trajs, epoch_len, 2)
    rooms, positions, actions = env.static_replay(initial_actions)
    # rooms_legacy, positions_legacy, actions_legacy = env_legacy.static_replay(initial_actions)
    # logging.critical('max room difference : {}'.format((rooms - rooms_legacy).max()))
    # logging.critical('max position difference : {}'.format((positions - positions_legacy).max()))

    cumulated_actions = np.cumsum(actions, axis=1)

    if resetting_type == 'fixed':
        reset_every = int(1/im_availability)
        ims_to_perturb = ((tch.tensor(range(epoch_len+1))-1).fmod(reset_every)!=0).unsqueeze(0).repeat((batch_size, 1)).float()
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

    all_errors_noiseless = np.zeros((n_trajs*n_seeds, epoch_len))
    all_errors_noiseless_no_images = np.zeros((n_trajs*n_seeds, epoch_len))

    all_errors_noisy = np.zeros((n_trajs*n_seeds, epoch_len))
    all_errors_noisy_no_images = np.zeros((n_trajs*n_seeds, epoch_len))

    with tch.set_grad_enabled(False):
        for seed in range(n_seeds):
            print('Starting seed {}'.format(start_seed+seed))
            # env = FixedRewardWorld(**all_params['env_params'], seed=start_seed+seed)
            env = FixedRewardWorld(**all_params['env_params'], seed=start_seed+seed)
            # env_legacy = LegacyWorld(**all_params['env_params'], seed=start_seed+seed)
            # if use_reimplementation is None:
            net = network_register[all_params['net_params']['net_name']](**all_params['net_params']['options'])
            # elif use_reimplementation == 'Small':
            #     net = ReimplementationPathIntegrator(**all_params['net_params']['options'])
            # elif use_reimplementation == 'Big':
            #     net = BigReimplementationPathIntegrator(**all_params['net_params']['options'])
            net.load_state_dict(tch.load(path+'seed{}/best_net.tch'.format(start_seed+seed)), strict=False)

            errors_noiseless = np.zeros((n_trajs, epoch_len))
            errors_noiseless_no_images = np.zeros((n_trajs, epoch_len))

            errors_noisy = np.zeros((n_trajs, epoch_len))
            errors_noisy_no_images = np.zeros((n_trajs, epoch_len))

            # First, do the noisy versions
            for batch_idx in range(n_trajs//batch_size):
                rooms_loc = deepcopy(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size])
                positions_loc = positions[batch_idx*batch_size:(batch_idx+1)*batch_size]
                # logging.critical('in aggregators, room shape : {} positions shape {}'.format(rooms_loc.shape, positions_loc.shape))
                # logging.critical('In aggregators : rooms {} ; positions {}'.format(rooms_loc[0, :3], positions_loc[0, :3]))

                images = env.get_images(rooms_loc, positions_loc)
                # images_legacy = env_legacy.get_images(rooms_loc, positions_loc)
                # images_legacy = env_legacy.get_images(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size], positions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                # logging.critical('max difference in images : {}'.format((images-images_legacy).abs().max()))
                # logging.critical('mean difference in images : {}'.format((images-images_legacy).abs().mean()))
                # logging.critical(images[:3] - images_legacy[:3])
                # logging.critical(images[:3] + images_legacy[:3])
                # logging.critical)

                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                # representations = net.get_representation(images_legacy.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)

                actions_encodings =  net.get_z_encoding(tch.from_numpy(actions[batch_idx*batch_size:(batch_idx+1)*batch_size]
                                                            + noise*np.random.randn(*actions[batch_idx*batch_size:(batch_idx+1)*batch_size].shape)).view(batch_size * (epoch_len), 2).float().to(net.device))
                actions_encodings = actions_encodings.view(batch_size, (epoch_len), -1)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                outputs, _, _ = net.do_path_integration(representations, actions_encodings)
                outputs = outputs.detach().cpu().numpy()
                errors_noisy[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))

                fully_perturbed_representations = deepcopy(representations)
                for t in tqdm(range(epoch_len)):
                    fully_perturbed_representations[:, t] = fully_perturbed_representations[:, t, tch.randperm(fully_perturbed_representations.shape[2])]
                fully_perturbed_representations[:, 0] = representations[:, 0]
                outputs_no_images, _, _ = net.do_path_integration(fully_perturbed_representations, actions_encodings)
                outputs_no_images = outputs_no_images.detach().cpu().numpy()
                errors_noisy_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs_no_images - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))

            # Then, the noiseless ones
            for batch_idx in range(n_trajs//batch_size):
                images = env.get_images(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size], positions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                # images_legacy = en    v_legacy.get_images(rooms[batch_idx*batch_size:(batch_idx+1)*batch_size], positions[batch_idx*batch_size:(batch_idx+1)*batch_size])
                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                # representations = net.get_representation(images_legacy.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)

                # logging.critical('max difference in images : {}'.format((images-images_legacy).abs().max()))
                # logging.critical('mean difference in images : {}'.format((images-images_legacy).abs().mean()))

                actions_encodings =  net.get_z_encoding(tch.from_numpy(actions[batch_idx*batch_size:(batch_idx+1)*batch_size]).view(batch_size * (epoch_len), 2).float().to(net.device))
                actions_encodings = actions_encodings.view(batch_size, (epoch_len), -1)
                representations = mask * representations
                tmp = representations[corrupt]
                tmp = tmp[:, tch.randperm(tmp.shape[1])]
                representations[corrupt] = tmp

                outputs, g, _ = net.do_path_integration(representations, actions_encodings)
                # print(g[:,:, 0].min(), g[:,:, 0].max(), g[:,:, 0].mean())
                outputs = outputs.detach().cpu().numpy()
                errors_noiseless[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))

                fully_perturbed_representations = deepcopy(representations)
                for t in tqdm(range(epoch_len)):
                    fully_perturbed_representations[:, t] = fully_perturbed_representations[:, t, tch.randperm(fully_perturbed_representations.shape[2])]
                # fully_perturbed_representations = fully_perturbed_representations[:, :, tch.randperm(fully_perturbed_representations.shape[2])]
                fully_perturbed_representations[:, 0] = representations[:, 0]
                outputs_no_images, g, _ = net.do_path_integration(fully_perturbed_representations, actions_encodings)
                # print(g[:,:, 0].min(), g[:,:, 0].max(), g[:,:, 0].mean())
                outputs_no_images = outputs_no_images.detach().cpu().numpy()
                errors_noiseless_no_images[batch_idx*batch_size:(batch_idx+1)*batch_size] = np.sqrt(((outputs_no_images - cumulated_actions[batch_idx*batch_size:(batch_idx+1)*batch_size])**2).sum(axis=-1))

            all_errors_noiseless[seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless
            all_errors_noiseless_no_images[seed*n_trajs:(seed+1)*n_trajs] = errors_noiseless_no_images

            all_errors_noisy[seed*n_trajs:(seed+1)*n_trajs] = errors_noisy
            all_errors_noisy_no_images[seed*n_trajs:(seed+1)*n_trajs] = errors_noisy_no_images

        all_errors_noiseless = np.reshape(all_errors_noiseless, (-1,) + all_errors_noiseless.shape[1:])
        all_errors_noiseless_no_images = np.reshape(all_errors_noiseless_no_images, (-1,) + all_errors_noiseless_no_images.shape[1:])
        all_errors_noisy = np.reshape(all_errors_noisy, (-1,) + all_errors_noisy.shape[1:])
        all_errors_noisy_no_images = np.reshape(all_errors_noisy_no_images, (-1,) + all_errors_noisy_no_images.shape[1:])

        if not collapsed:
            fig = plt.figure(tight_layout=True, figsize=(10, 5))
            gs = matplotlib.gridspec.GridSpec(2, 2)

            ax = fig.add_subplot(gs[0, 0])
            plot_mean_std(ax, all_errors_noiseless_no_images)
            ax.set_title('Without images', fontsize=20)
            ax.set_ylabel(r"\begin{center}Perfect\\reafference\end{center}", fontsize=20)

            ax = fig.add_subplot(gs[0, 1])
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k')
            plot_mean_std(ax, all_errors_noiseless)
            ax.set_title('With images', fontsize=20)

            ax = fig.add_subplot(gs[1, 0])
            plot_mean_std(ax, all_errors_noisy_no_images)
            ax.set_ylabel(r'\begin{center}Noisy\\reafference\end{center}', fontsize=20)

            ax = fig.add_subplot(gs[1, 1])
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k')
            plot_mean_std(ax, all_errors_noisy)

            fig.savefig(path+'resetting_errors_summary_im_availability_{}_noise_{}_resetting_{}.pdf'.format(im_availability, noise, resetting_type))
            plt.close(fig)
        else:
            fig, axes = plt.subplots(1,2, figsize=(10,3.5))

            # ax = axes[0]
            # plot_mean_std(ax, all_errors_noiseless_no_images, label=r'$\epsilon=0$', c_line='tab:blue', c_fill='tab:blue')
            # plot_mean_std(ax, all_errors_noisy_no_images, label=r'$\epsilon=0.075$', c_line='tab:orange', c_fill='tab:orange')
            # ax.legend()
            # ax.set_title('Without images', fontsize=20)
            #
            # ax = axes[1]
            # for t in range(1, epoch_len):
            #     if ims_to_perturb[0, t+1] == 0:
            #         ax.axvline(x=t, ls='--', c='k')
            # plot_mean_std(ax, all_errors_noiseless, label=r'$\epsilon=0$', c_line='tab:blue', c_fill='tab:blue')
            # plot_mean_std(ax, all_errors_noisy, label=r'$\epsilon=0.075$', c_line='tab:orange', c_fill='tab:orange')
            # ax.legend()
            # ax.set_title('With images', fontsize=20)


            ax = axes[0]
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k', zorder=5, alpha=.5)
            plot_mean_std(ax, all_errors_noiseless_no_images, label=r'Without images', c_line='tab:blue', c_fill='tab:blue')
            plot_mean_std(ax, all_errors_noiseless, label=r'With images', c_line='tab:orange', c_fill='tab:orange')
            ax.set_xlabel('Time')
            ax.set_ylabel('PI error')
            ax.legend(loc=2, framealpha=1)
            ax.set_title(r'Perfect reafference ($\epsilon=0$)')

            ax = axes[1]
            for t in range(1, epoch_len):
                if ims_to_perturb[0, t+1] == 0:
                    ax.axvline(x=t, ls='--', c='k', zorder=5, alpha=.5)
            plot_mean_std(ax, all_errors_noisy_no_images, label=r'Without images', c_line='tab:blue', c_fill='tab:blue')
            plot_mean_std(ax, all_errors_noisy, label=r'With images', c_line='tab:orange', c_fill='tab:orange')
            ax.set_xlabel('Time')
            ax.set_ylabel('PI error')
            ax.legend(loc=2, framealpha=1)
            ax.set_title(r'Noisy reafference ($\epsilon=0.075$)')

            fig.savefig(path+'collapsed_resetting_errors_summary_im_availability_{}_noise_{}_resetting_{}.pdf'.format(im_availability, noise, resetting_type))
            plt.close(fig)


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

if __name__ == '__main__':
    offline_study_errors(map='SnakePath', layout='Default', exp_group='end_to_end/', protocol='default', epoch_len=100, step_size=.5, batch_size=128, n_trajs=512, start_seed=0, n_seeds=2, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
    offline_study_errors(map='SnakePath', layout='Default', exp_group='end_to_end/', protocol='no_fb_losses', epoch_len=100, step_size=.5, batch_size=128, n_trajs=512, start_seed=0, n_seeds=2, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
    # offline_study_errors(map='SnakePath', layout='Default', exp_group='end_to_end/reimplementation_retrained', protocol='GRU_epoch_len_15', epoch_len=100, step_size=.5, batch_size=128, n_trajs=512, start_seed=0, n_seeds=2, im_availability=.1, corruption_rate=.5, noise=0.05, resetting_type='fixed', collapsed=True)
