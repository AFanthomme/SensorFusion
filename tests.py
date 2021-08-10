# These are just the visual tests, meant mostly for disagnostics
# Main figures (aggregated across seeds for example) are in aggregator.py

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
from itertools import cycle
from scipy.signal import lfilter
from environment import meaningful_trajectories
from policy import policy_register

# tests_register = {}
max = lambda x, y: x if x > y else y
min = lambda x, y: x if x < y else y

def __add_arrows(line, size=15, color=None, zorder=-1):
    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    x_ends = .5* (xdata[:-1] + xdata[1:])
    y_ends = .5* (ydata[:-1] + ydata[1:])
    for x_start, x_end, y_start, y_end in zip(xdata, x_ends, ydata, y_ends):
        line.axes.annotate('',
            xytext=(x_start, y_start),
            xy=(x_end, y_end),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size, zorder=-1
        )

def sanity_check_position(net, env, pars, epoch):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    with tch.set_grad_enabled(False):
        n_rooms = env.n_rooms
        resolution = pars['resolution']
        os.makedirs(net.save_folder + '{}/'.format(epoch) + 'sanity_check_position', exist_ok=True)

        x_room = np.linspace(-env.scale, env.scale, resolution)
        y_room = np.linspace(-env.scale, env.scale, resolution)
        xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])

        # EXPAND: Add new environment here
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

        all_images = env.get_images(rooms, xy_local)
        start_images = tch.cat([env.get_images(start_room, start_pos) for start_room, start_pos in start_pos_list], dim=0)
        all_reps = net.get_representation(all_images)

        start_pos = start_pos_list[0][1]
        assert tch.sum((env.get_images(np.array([0]), start_pos)- env.get_images(np.array([1]), start_pos))**2) > 0
        assert tch.sum((env.get_images(np.array([3]), start_pos)- env.get_images(np.array([4]), start_pos))**2) > 0
        assert tch.sum((env.get_images(np.array([2]), start_pos)- env.get_images(np.array([4]), start_pos))**2) > 0

        start_reps = net.get_representation(start_images)

        for start_point_idx, start_rep in enumerate(start_reps):
            start_im = start_images[start_point_idx]
            start_label = start_labels[start_point_idx]
            start_room, start_pos = start_pos_list[start_point_idx]
            delta_r = xy_global - env.room_centers[start_room, :2] - start_pos

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            seismic = plt.get_cmap('seismic')
            reds = plt.get_cmap('Reds')
            dists = net.backward_model(tch.cat([start_rep.reshape([1, -1]) for _ in range(all_reps.shape[0])], dim=0), all_reps).detach().cpu().numpy()
            log_errors = np.log10(np.sqrt(((dists-delta_r)**2).sum(axis=-1)))

            tmp = max(np.abs(delta_r.min()), np.abs(delta_r.max()))
            norm = matplotlib.colors.Normalize(vmin=-tmp, vmax=tmp)

            ax = axes[0]
            ax = env.render_template(ax_to_use=ax)
            patch = Circle((env.room_centers[start_room,0] + start_pos[0], env.room_centers[start_room,1] + start_pos[1]), .1 * env.scale, linewidth=1, edgecolor='k', facecolor=[0,0,0,.2])
            ax.add_patch(patch)
            ax.set_title('Reconstructed x from anchor point')
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm(dists[:,0].flatten())), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
            fig.add_axes(ax_cb)

            ax = axes[1]
            ax = env.render_template(ax_to_use=ax)
            patch = Circle((env.room_centers[start_room,0] + start_pos[0], env.room_centers[start_room,1] + start_pos[1]), .1 * env.scale, linewidth=1, edgecolor='k', facecolor=[0,0,0,.2])
            ax.add_patch(patch)
            ax.set_title('Reconstructed y from anchor point')
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm(dists[:,1].flatten())), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm, orientation='vertical')
            fig.add_axes(ax_cb)

            norm_log_error = matplotlib.colors.Normalize(vmin=log_errors.min(), vmax=log_errors.max())
            ax = axes[2]
            ax = env.render_template(ax_to_use=ax)
            patch = Circle((env.room_centers[start_room,0] + start_pos[0], env.room_centers[start_room,1] + start_pos[1]), .1 * env.scale, linewidth=1, edgecolor='k', facecolor=[0,0,0,.2])
            ax.add_patch(patch)
            ax.set_title('Reconstruction error from anchor point (base 10 log)')
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm_log_error(log_errors.flatten())), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm_log_error, orientation='vertical')
            fig.add_axes(ax_cb)

            fig.tight_layout()
            fig.savefig(net.save_folder + '{}/'.format(epoch) + 'sanity_check_position/{}.pdf'.format(start_label))
            plt.close(fig)


def sanity_check_representation(net, env, pars, epoch):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    with tch.set_grad_enabled(False):
        n_rooms = env.n_rooms
        resolution = pars['resolution']
        os.makedirs(net.save_folder + '{}/'.format(epoch) + 'sanity_check_representation', exist_ok=True)

        x_room = np.linspace(-env.scale, env.scale, resolution)
        y_room = np.linspace(-env.scale, env.scale, resolution)
        xy_room = np.transpose([np.tile(x_room, len(y_room)), np.repeat(y_room, len(x_room))])

        # EXPAND: Add new environment here
        if env.map_name in ['BigRoom']:
            rooms = np.zeros(resolution**2).astype(int)
            x_global = x_room
            y_global = y_room
            xy_global = xy_room
            xy_local = xy_room
        elif env.map_name in ['DonutPath', 'SnakePath', 'DoubleDonut']:
            rooms = np.concatenate([[room_idx]*(resolution**2) for room_idx in range(env.n_rooms)], axis=0)
            logging.critical(np.bincount(rooms))
            x = xy_room[:, 0]
            y = xy_room[:, 1]

            x_global = np.concatenate([x+env.room_centers[room_idx, 0] for room_idx in range(env.n_rooms)], axis=0)
            y_global = np.concatenate([y+env.room_centers[room_idx, 1] for room_idx in range(env.n_rooms)], axis=0)
            xy_global = np.concatenate([x_global.reshape(-1, 1), y_global.reshape(-1, 1)], axis=1)

            x_local = np.concatenate([x for room_idx in range(env.n_rooms)], axis=0)
            y_local = np.concatenate([y for room_idx in range(env.n_rooms)], axis=0)
            xy_local = np.concatenate([x_local.reshape(-1, 1), y_local.reshape(-1, 1)], axis=1)

        all_images = env.get_images(rooms, xy_local)
        all_reps = net.get_representation(all_images).cpu().numpy()

        seismic = plt.get_cmap('seismic')
        for neuron in range(16):
            fig, ax = plt.subplots()
            ax = env.render_template(ax_to_use=ax)
            norm_rep = matplotlib.colors.Normalize(vmin=all_reps[:,neuron].min(), vmax=all_reps[:,neuron].max())
            ax.scatter(xy_global[:,0].flatten(), xy_global[:,1].flatten(), c=seismic(norm_rep(all_reps[:,neuron])), s=64000/(resolution**2), rasterized=True, zorder=-5)
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=seismic, norm=norm_rep, orientation='vertical')
            fig.add_axes(ax_cb)
            fig.tight_layout()
            fig.savefig(net.save_folder + '{}/'.format(epoch) + 'sanity_check_representation/rep_{}.pdf'.format(neuron))
            plt.close(fig)


def path_integrator_test(net, env, pars, epoch):
    with tch.set_grad_enabled(False):
        os.makedirs(net.save_folder + '{}/'.format(epoch) + 'path_integration_tests/', exist_ok=True)
        epoch_len = pars['epoch_len']
        step_size = pars['step_size']
        im_availability = pars['im_availability']
        corruption_rate = pars['corruption_rate']
        policy_pars = deepcopy(pars['policy_pars']) # Do a copy because we want to act on it without causing side-effects
        policy_pars['seed'] = 777
        batch_size = 5

        rng = np.random.RandomState(777)
        # start_rooms = rng.choice(env.n_rooms, size=(100,))
        start_rooms = rng.choice(env.possible_start_rooms, size=(100,))
        start_pos = rng.uniform(-env.scale, env.scale, size=(100, 2))

        for action_type in ['random', 'semi_deliberate']:
            for resetting_positions_type in ['random', 'fixed']:
                if action_type == 'random':
                    policy = policy_register[policy_pars['type']](**policy_pars)
                    actions = policy.get_batch_of_actions(batch_size=batch_size, epoch_len=epoch_len)

                elif action_type == 'semi_deliberate':
                    logging.critical('Working on semi deliberate actions')
                    try:
                        deliberate_actions = np.array([meaningful_trajectories[env.map_name]]*batch_size) * env.scale
                        rooms, positions, _ = env.static_replay(deliberate_actions, start_rooms=np.zeros(batch_size, dtype=int), start_pos=np.zeros((batch_size, 2)))

                        perturbed_positions = np.clip(positions+.2*env.scale*np.random.uniform(-1, 1, size=positions.shape), -env.scale, env.scale)
                        global_positions = perturbed_positions + env.room_centers[rooms.astype(int)][:,:,:2]
                        actions = global_positions[:, 1:] - global_positions[:, :-1]

                        actions[0] = deliberate_actions[0]
                        epoch_len = actions.shape[1]
                        start_rooms = np.zeros(batch_size)
                        start_pos = np.zeros((batch_size, 2))
                    except Exception as e:
                        print(e)
                        continue

                rooms, positions, actions = env.static_replay(actions, start_rooms=start_rooms, start_pos=start_pos)
                outputs = np.zeros((batch_size, epoch_len, 2))
                gatings = np.zeros((batch_size, epoch_len))
                outputs_no_visual = np.zeros((batch_size, epoch_len, 2))

                cumulated_actions = np.cumsum(actions, axis=1)
                images = env.get_images(rooms, positions) #retinal states, (bs, T+1=2, ret_res**2, 3)
                representations = net.get_representation(images.view(batch_size * (epoch_len+1), -1, 3)).view(batch_size, (epoch_len+1), -1)
                actions_encodings =  net.get_z_encoding(tch.from_numpy(actions).view(batch_size * (epoch_len), 2).float().to(net.device)).view(batch_size, (epoch_len), -1)

                fully_perturbed_representations = deepcopy(representations)

                # Single permutation for all neurons, should be fine
                fully_perturbed_representations = fully_perturbed_representations[:, :, tch.randperm(fully_perturbed_representations.shape[2])]
                fully_perturbed_representations[:, 0] = representations[:, 0]
                outputs_no_visual, _, _ = net.do_path_integration(fully_perturbed_representations, actions_encodings)
                outputs_no_visual = outputs_no_visual.detach().cpu().numpy()

                if resetting_positions_type == 'random':
                    ims_to_perturb =  tch.bernoulli((1.-im_availability) * tch.ones(batch_size, epoch_len+1))
                    # logging.critical('in random : ims_to_perturb : min {}, max {}, mean {}'.format(ims_to_perturb.min(), ims_to_perturb.max(), ims_to_perturb.mean()))
                elif resetting_positions_type == 'fixed':
                    reset_every = int(1/im_availability)
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

                outputs, gatings, internal_states = net.do_path_integration(representations, actions_encodings)

                outputs = outputs.detach().cpu().numpy()
                gatings = gatings.detach().cpu().numpy()

                time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[1]+1)
                cmap = plt.get_cmap('jet')
                colors = cmap(time_based_norm(range(epoch_len+1)))

                for b in range(batch_size):
                    if env.map_name != 'DoubleDonut':
                        fig, axes = plt.subplots(1, 2, figsize=(10,5))
                    elif env.map_name == 'DoubleDonut':
                        fig, axes = plt.subplots(1, 2, figsize=(20,5))

                    ax = axes[0]
                    ax = env.render_template(ax_to_use=ax)

                    global_pos_with_reset  = np.zeros((epoch_len+1, 2))
                    global_pos_with_reset[0] = env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                    global_pos_with_reset[1:] = outputs[b] + env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate

                    global_pos_without_reset  = np.zeros((epoch_len+1, 2))
                    global_pos_without_reset[0] = env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                    global_pos_without_reset[1:] = outputs_no_visual[b] + env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate

                    true_global_pos  = np.zeros((epoch_len+1, 2))
                    true_global_pos[0] = env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate
                    true_global_pos[1:] = cumulated_actions[b] + env.room_centers[int(rooms[b,0]), :2] + positions[b,0] # NOTE: :2 is there to prepare arival of "z" coordinate

                    for t in range(epoch_len - 1):
                        marker_true = '+'
                        marker_with_reset = '*'
                        marker_without_reset = '.'
                        ax.scatter(global_pos_with_reset[t,0], global_pos_with_reset[t,1], c=colors[t], marker=marker_with_reset, alpha=.5, zorder=-5)
                        # ax.scatter(global_pos_without_reset[t,0], global_pos_without_reset[t,1], c=colors[t], marker=marker_without_reset, alpha=.7) # That makes the figure too confusing
                        ax.scatter(true_global_pos[t,0], true_global_pos[t,1], c=colors[t], marker=marker_true, alpha=.5, zorder=-5)

                    divider = make_axes_locatable(ax)
                    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=time_based_norm, orientation='vertical')
                    fig.add_axes(ax_cb)
                    ax.set_title('Recovered positions')

                    ax = axes[1]
                    ax.plot(gatings[b, :epoch_len, 0])
                    ax.axhline(y=0, c='k')
                    for t in range(epoch_len - 1):
                        if ims_to_perturb[b, t+1] == 0:
                            ax.axvline(x=t, ls='--', c='k')
                    ax.set_title('Value of the gating')

                    fig.tight_layout()
                    fig.savefig(net.save_folder + '{}/'.format(epoch) + 'path_integration_tests/{}_traj{}_resetting_{}.pdf'.format(action_type, b, resetting_positions_type))
                    plt.close('all')





tests_register = {'sanity_check_position': sanity_check_position, 'sanity_check_representation': sanity_check_representation, 'path_integrator_test': path_integrator_test}
