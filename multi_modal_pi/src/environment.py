import numpy as np
from copy import deepcopy
from numpy.random import RandomState
import logging
import torch as tch
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
import os
import gym
from gym.utils import seeding
from gym import spaces
import json


from src.networks import *
from src.retina import Retina

from maps.SnakePath import snakepath_layout_dict, snakepath_trajectory
from maps.DoubleDonut import doubledonut_layout_dict, doubledonut_trajectory

environment_register = {
    'SnakePath': snakepath_layout_dict,
    'DoubleDonut': snakepath_layout_dict,
}

# These are made to use steps of size .5*env.scale, ie 1/4 of a room each time
meaningful_trajectories = {
    'SnakePath': snakepath_trajectory,
    'DoubleDonut': doubledonut_trajectory,
}


class World:
    def __init__(self, map_name='SnakePath', scale=.5, reward_area_width=.3, objects_layout_name='Default', epoch_len=20, seed=0, load_preprocessor_from=None, **kwargs):
        self.epoch_len = epoch_len
        self.retina = Retina(n=64**2, bounds=[-.5,.5], widths=[.3, .5], device_name='cuda')
        self.device = self.retina.device
        self.map_name = map_name
        self.objects_layout_name = objects_layout_name

        env_params = deepcopy(environment_register[map_name])
        self.env_params = env_params
        self.n_rooms = len(env_params['room_centers'])
        logging.critical('n_rooms in our env : {}'.format(self.n_rooms))
        self.room_labels = ["Room {}".format(i) for i in range(self.n_rooms)]
        self.room_centers = env_params['room_centers']
        self.room_sizes = env_params['room_sizes']
        self.room_exits = env_params['room_exits']
        self.room_objects = env_params['possible_layouts'][objects_layout_name]
        self.scale = scale

        self.max_objects_per_room = np.max([objs_dict['positions'].shape[0] for objs_dict in self.room_objects])
        self.colors_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 3))
        self.positions_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 2))
        for room_idx in range(self.n_rooms):
            for obj_idx in range(self.room_objects[room_idx]['colors'].shape[0]):
                self.colors_blob[room_idx, obj_idx] = self.room_objects[room_idx]['colors'][obj_idx]
                self.positions_blob[room_idx, obj_idx] = self.room_objects[room_idx]['positions'][obj_idx]

        self.room_sizes = scale * self.room_sizes
        self.room_centers = scale * self.room_centers

        for i in range(self.n_rooms):
            self.room_objects[i]['positions'] = scale * self.room_objects[i]['positions']
            for obj_idx in range(len(self.room_exits[i])):
                self.room_exits[i][obj_idx]['goes_to'][1] = scale * np.array(self.room_exits[i][obj_idx]['goes_to'][1])
                self.room_exits[i][obj_idx]['x'] = scale * self.room_exits[i][obj_idx]['x']
                self.room_exits[i][obj_idx]['y'] = scale * self.room_exits[i][obj_idx]['y']

        # Useful for losses; for now, only rooms with no objects
        self.rooms_not_to_start_in = [i for i in range(self.n_rooms) if np.all(self.room_objects[i]['colors'] == 0.)]
        self.possible_start_rooms = [i for i in range(self.n_rooms) if not np.all(self.room_objects[i]['colors'] == 0.)]

        if self.map_name == 'DoubleDonut' and self.objects_layout_name == 'Ambiguous':
             self.possible_start_rooms = np.array([0,1,2,3,4,5,6,8,9,10,12,13,14,15]) # Do not allow start from ambiguous rooms

        logging.critical('Allowed starting rooms : {}'.format(self.possible_start_rooms))
        self.set_seed(seed)

        self.seed_value = seed
        self.reward_room = None

        if load_preprocessor_from is not None:
            with open(load_preprocessor_from+'/full_params.json', mode='r') as f:
                plop = json.load(f)
                net_params = plop['net_params']['options']
                net_name = plop['net_params']['net_name']
            self.preprocessor = network_register[net_name](**net_params).to(self.device)
            logging.critical('Attempting to load preprocessor from : {}'.format(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value)))
            self.preprocessor.load(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value))
            if not kwargs['use_recurrence']:
                self.state_size = (self.preprocessor.representation_size)
            else:
                self.state_size = (self.preprocessor.hidden_size)

        else:
            self.preprocessor = None

    def set_seed(self, seed=None):
        logging.critical('called world.set_seed')
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_background(self, ax_to_use=None):
        if ax_to_use is None:
            fig, ax = plt.subplots()
        else:
            ax = ax_to_use

        ax.set_facecolor([0,0,0,0])
        ax.set_axis_off()
        ax.set_aspect('equal')

        ax.set_xlim([np.min(self.room_centers[:,0])-self.scale-.05, np.max(self.room_centers[:,0])+self.scale+.05])
        ax.set_ylim([np.min(self.room_centers[:,1])-self.scale-.05, np.max(self.room_centers[:,1])+self.scale+.05])

        # This one is common to all environments, it determines the enclosing area
        rect = patches.Rectangle((np.min(self.room_centers[:,0])-self.scale, np.min(self.room_centers[:,1]-self.scale)),
                                np.max(self.room_centers[:,0]) - np.min(self.room_centers[:,0])+ 2*self.scale,
                                np.max(self.room_centers[:,1]) - np.min(self.room_centers[:,1])+ 2*self.scale,
                                linewidth=1, edgecolor='k',facecolor=[0,0,0,0])
        ax.add_patch(rect)

        if self.map_name == 'SnakePath':
            ax.plot([-3 *self.scale, self.scale], [-self.scale, -self.scale], c='k', linewidth=3)
            ax.plot([-self.scale, 3*self.scale], [self.scale, self.scale], c='k', linewidth=3)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DonutPath':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)
            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
        elif self.map_name == 'DoubleDonut':
            rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

            rect = patches.Rectangle((5*self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
            ax.add_patch(rect)

            ax.plot([3*self.scale, 3*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [-self.scale, -self.scale], c='k', ls='--')
            ax.plot([3*self.scale, 9*self.scale], [self.scale, self.scale], c='k', ls='--')
            ax.plot([5*self.scale, 5*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
            ax.plot([7*self.scale, 7*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')

        if ax_to_use is None:
            return fig, ax
        else:
            return ax

    def get_observation(self, room, position, action=None, is_reset=False):
        # is_reset is useless here, but will be for RNN preprocessor
        position = tch.from_numpy(self.positions_blob[room] - position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0)).float()
        color = tch.from_numpy(self.colors_blob[room]).float().to(self.retina.device)
        image_batch = self.retina.activity(position, color)
        image_batch = image_batch.squeeze(0).detach().cpu().numpy()
        return image_batch

    def get_images(self, rooms, positions):
        desired_shape = deepcopy(rooms.shape)
        rooms = rooms.flatten().astype(int)
        positions = positions.reshape(-1, 2)
        # logging.critical('positions info : shape {}, type {}'.format(positions.shape, positions.dtype))
        all_positions = tch.from_numpy(self.positions_blob[rooms] - positions[:, np.newaxis, :].repeat(self.max_objects_per_room, axis=1)).float()

        all_positions = all_positions.to(self.retina.device)
        all_colors = tch.from_numpy(self.colors_blob[rooms]).float().to(self.retina.device)
        # logging.critical('In new World : allpositions {} ; all_colors {}'.format(all_positions[:3], all_colors[:3]))
        image_batch = self.retina.activity(all_positions, all_colors)

        return image_batch.reshape(*desired_shape, self.retina.n, 3)#.cpu().numpy()

    def check_reward_overlap(self, room, pos):
        if self.reward_room is not None:
            overlaps = np.logical_and(room == self.reward_room, np.max(np.abs(pos-self.reward_position))<=self.reward_area_width)
            return overlaps
        else:
            return False


    # Below that are functions used for legacy interface
    def set_agent_position(self, room, xy=(0,0)):
        self.agent_room = room
        Lx, Ly = self.room_sizes[self.agent_room]
        invalid_x = xy[0]>Lx or xy[0]<-Lx
        invalid_y = xy[1]>Ly or xy[1]<-Ly
        if invalid_x or invalid_y:
            raise RuntimeError('Invalid xy initialization for current room')
        self.agent_position = np.array([*xy])

    def render_template(self, ax_to_use=None, add_goal=False):
        if ax_to_use is None:
            fig, ax = self.render_background(ax_to_use=None)
        else:
            ax = self.render_background(ax_to_use=ax_to_use)

        for room in range(self.n_rooms):
            for xy, col in zip(self.room_objects[room]['positions'], self.room_objects[room]['colors']):
                xy0 = self.room_centers[room, :2]
                rect = patches.Rectangle(xy+xy0-.05*self.scale, .2*self.scale, .2*self.scale, linewidth=1, edgecolor='k', facecolor=col)
                ax.add_patch(rect)

        if add_goal:
            if self.reward_room is not None:
                exit_rect = patches.Rectangle(self.room_centers[self.reward_room, :2]+self.reward_position-self.reward_area_width, 2*self.reward_area_width, 2*self.reward_area_width, facecolor='gray', hatch='x', alpha=.5)
                ax.add_patch(exit_rect)

        if ax_to_use is None:
            return fig, ax
        else:
            return ax

    def __replay_one_traj(self, actions, start_room=None, start_pos=None, reward_room=None, reward_pos=None):
        self.reset()
        epoch_len = actions.shape[0]
        positions = np.zeros((epoch_len+1,2))
        validated_actions = np.zeros((epoch_len,2))
        rooms = np.zeros((epoch_len+1))

        if start_room is None:
            room, pos = self.agent_room, self.agent_position
        else:
            room, pos = start_room, start_pos
            room = int(room)
            self.set_agent_position(room, (pos[0], pos[1]))
            self.reward_room = reward_room
            self.reward_position = reward_pos

        positions[0] = pos
        rooms[0] = room

        for idx, action in enumerate(actions):
            # new_room, new_pos, rectified_action, reward, is_terminal = self.step(action)
            obs, reward, end_traj, info = self.step(action)
            new_room, new_pos, rectified_action = info['new_room'], info['new_pos'], info['rectified_action']
            logging.debug('Start in room {} at ({},{}) and end in room {} at ({},{}) with tot_dep ({},{})'.format(room, *pos, new_room, *new_pos, *rectified_action))
            validated_actions[idx] = rectified_action
            positions[idx+1] = new_pos
            rooms[idx+1] = new_room
            pos = new_pos
            room = new_room

            if end_traj:
                positions[idx+1:] = new_pos
                rooms[idx+1:] = new_room
                validated_actions[idx+1:] = 0.
                break
        return rooms, positions, validated_actions


    def static_replay(self, actions_batch, start_rooms=None, start_pos=None):
        actions_batch_local = deepcopy(actions_batch)
        batch_size = actions_batch.shape[0]
        epoch_len = actions_batch.shape[1]

        if start_rooms is not None:
            assert start_pos is not None
            assert start_rooms.shape[0] == start_pos.shape[0]
            assert start_pos.shape[1] == 2

        rooms = np.zeros((batch_size, epoch_len+1))
        positions = np.zeros((batch_size, epoch_len + 1, 2))
        validated_actions = np.zeros((batch_size, epoch_len, 2))
        rewards = np.zeros((batch_size, epoch_len+1))
        irrelevant_times = np.zeros((batch_size, epoch_len+1))

        # NOTE: making this multi-threaded seems smart, but at least in my tests its either slow, buggy, or both
        for b in range(batch_size):
            if start_rooms is None:
                room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
            else:
                room, pos, act, =  self.__replay_one_traj(actions_batch_local[b], start_room=start_rooms[b], start_pos=start_pos[b])

            rooms[b, :] = room
            positions[b, :] = pos
            validated_actions[b, :] = act

        logging.debug('Done with static_replay in environments.py')
        return rooms, positions, validated_actions


    @staticmethod
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


    def plot_trajectory(self, actions, start_room=None, start_pos=None, reward_room=None, reward_pos=None, ax_to_use=None, save_file=None, color=None, marker='+', zorder=500, show_lines=False, show_arrows=False, s=32, **kwargs):
        # By default, color is set to show time, but can be overridden
        if color is None:
            time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[0]+1)
            cmap = plt.get_cmap('jet')
            color = cmap(time_based_norm(range(actions.shape[0]+1)))

        if ax_to_use is None:
            if self.map_name != 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(5,5))
            elif self.map_name == 'DoubleDonut':
                fig, ax = plt.subplots(figsize=(10,5))

        else:
            ax = ax_to_use

        # if start_room is None:
        #     room, pos, act =  self.__replay_one_traj(actions, start_room=None, start_pos=None)
        # else:
        room, pos, act =  self.__replay_one_traj(actions, start_room=start_room, start_pos=start_pos, reward_room=reward_room, reward_pos=reward_pos)

        ax = self.render_template(ax_to_use=ax)
        absolute_pos = pos+self.room_centers[room.astype(int), :2]

        ax.scatter(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, marker=marker, s=s)

        if show_lines:
            lines = ax.plot(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, ls='-')
            if show_arrows:
                self.__add_arrows(lines[0], zorder=zorder, color=color)

        if save_file is not None:
            os.makedirs('/'.join(save_file.split('/')[:-1]), exist_ok=True)
            fig.savefig(save_file)
            plt.close(fig)
        elif kwargs['return_fig']:
            return fig
        else:
            return ax

    def movement_logic(self, action):
        assert len(action) == 2
        action_bkp = deepcopy(action)
        bkp_x, bkp_y = deepcopy(self.agent_position)
        room_bkp = deepcopy(self.agent_room)

        new_pos = np.array([self.agent_position[0] + action[0], self.agent_position[1] + action[1]])
        Lx, Ly = self.room_sizes[self.agent_room]

        invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx
        invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly

        if invalid_x and invalid_y:
            if self.np_random.uniform() < .5:
                action[0] -= new_pos[0] - np.clip(new_pos[0], -Lx, Lx)
                new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
                invalid_x = False
            else:
                action[1] -= new_pos[1] - np.clip(new_pos[1], -Ly, Ly)
                new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
                invalid_y = False


        changed_room = False
        if invalid_y:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='horizontal':
                    if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
                        logging.debug('crossed horizontal door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if invalid_x:
            for exit in self.room_exits[self.agent_room]:
                if changed_room:
                    break
                if exit['axis'] =='vertical':
                    if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
                        logging.debug('crossed vertical door')
                        used_exit = deepcopy(exit)
                        changed_room = True

        if not changed_room:
            new_room = room_bkp
        else:
            new_room = used_exit['goes_to'][0]
            new_pos = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[new_room, :2]

        rectified_new_pos = np.zeros(2)
        rectified_new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
        rectified_new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
        rectified_action = action + rectified_new_pos - new_pos

        self.agent_room = deepcopy(new_room)
        self.agent_position = deepcopy(rectified_new_pos)

        self.t += 1
        return self.agent_room, self.agent_position, deepcopy(rectified_action)

# To allow movement between rooms that touch only on one corner.
        # if not (invalid_x and invalid_y):
        #     changed_room = False
        #     if invalid_y:
        #         for exit in self.room_exits[self.agent_room]:
        #             if changed_room:
        #                 break
        #             if exit['axis'] =='horizontal':
        #                 if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
        #                     logging.debug('crossed horizontal door')
        #                     used_exit = deepcopy(exit)
        #                     changed_room = True
        #
        #     if invalid_x:
        #         for exit in self.room_exits[self.agent_room]:
        #             if changed_room:
        #                 break
        #             if exit['axis'] =='vertical':
        #                 if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
        #                     logging.debug('crossed vertical door')
        #                     used_exit = deepcopy(exit)
        #                     changed_room = True
        #
        #     if not changed_room:
        #         new_room = room_bkp
        #     else:
        #         new_room = used_exit['goes_to'][0]
        #         new_pos = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[new_room, :2]
        #         self.agent_room = deepcopy(new_room)
        #
        #
        # else:
        #     # Everything assumes no action can cross three boundaries (more than one in a direction)
        #     logging.critical('reached the two invalids branch')
        #
        #     crossings_are_done = False
        #
        #     while (invalid_x or invalid_y):
        #         changed_room = False
        #         logging.critical('started another round in two invalids branch')
        #         logging.critical('{} {} {}'.format(self.agent_room, self.agent_position,  new_pos))
        #
        #         if invalid_y:
        #             for exit in self.room_exits[self.agent_room]:
        #                 if changed_room:
        #                     break
        #                 if exit['axis'] =='horizontal':
        #                     logging.critical('{} {}'.format(np.clip(new_pos[1], -Ly, Ly), exit['y']))
        #                     if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
        #                         logging.critical('crossed horizontal door')
        #                         used_exit = deepcopy(exit)
        #                         changed_room = True
        #
        #         if changed_room:
        #             self.agent_room = used_exit['goes_to'][0]
        #             self.agent_position = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[self.agent_room, :2]
        #             # remaining_action = action_bkp - self.room_centers[room_bkp, :2] + self.room_centers[self.agent_room, :2]
        #             # new_pos = np.array([self.agent_position[0] + remaining_action[0], self.agent_position[1] + remaining_action[1]])
        #             room_bkp = deepcopy(self.agent_room)
        #             new_pos = self.agent_position
        #             Lx, Ly = self.room_sizes[self.agent_room]
        #             invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx
        #             invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly # Should never be true since we crossed one vertical
        #             logging.critical('{} {} {}'.format(self.agent_room, self.agent_position,  new_pos))
        #             continue
        #
        #         if invalid_x:
        #             for exit in self.room_exits[self.agent_room]:
        #                 if changed_room:
        #                     break
        #                 if exit['axis'] =='vertical':
        #                     if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
        #                         logging.critical('crossed vertical door')
        #                         used_exit = deepcopy(exit)
        #                         changed_room = True
        #
        #         if changed_room:
        #             logging.critical('reached changed_room block')
        #             self.agent_room = used_exit['goes_to'][0]
        #             self.agent_position = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[self.agent_room, :2]
        #             # remaining_action = action_bkp - self.room_centers[room_bkp, :2] + self.room_centers[self.agent_room, :2]
        #             room_bkp = deepcopy(self.agent_room)
        #             new_pos = self.agent_position
        #             Lx, Ly = self.room_sizes[self.agent_room]
        #             invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx # Should never be true since we crossed one horizontal
        #             invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly
        #             logging.critical('{} {} {} {} {}'.format(self.agent_room, self.agent_position,  new_pos, invalid_x, invalid_y))
        #             continue
        #
        #         break
        #
        # rectified_new_pos = np.zeros(2)
        # rectified_new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
        # rectified_new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
        # rectified_action = action + rectified_new_pos - new_pos
        #
        #
        # self.agent_position = deepcopy(rectified_new_pos)
        #
        # self.t += 1
        # return self.agent_room, self.agent_position, deepcopy(rectified_action)




class FixedRewardWorld(gym.Env, World):
    def __init__(self, scale=.5, reward_area_width=.3, chosen_reward_pos='Default', epoch_len=100, skip_reset=False, **kwargs):
        # print(kwargs)
        World.__init__(self, **kwargs)
        logging.critical(kwargs)

        self.epoch_len = epoch_len
        self.reward_room = self.env_params['possible_reward_pos'][chosen_reward_pos]['room']
        self.reward_position = np.array(self.env_params['possible_reward_pos'][chosen_reward_pos]['pos']) * self.scale
        self.reward_area_width = reward_area_width * self.scale

        # Gym specific stuff
        self.action_space = gym.spaces.Box(low=.8*self.scale *np.array([-1.0, -1.0]), high=.8*self.scale *np.array([1.0, 1.0]), dtype=np.float32) # Continuous actions, bounded for simplicity
        if self.preprocessor is None:
            self.observation_shape = (64**2, 3)
        else:
            self.observation_shape = (self.state_size,)

        self.observation_space = spaces.Box(low = -10 * np.ones(self.observation_shape),
                                            high = 10 * np.ones(self.observation_shape),
                                            dtype = np.float32)
        if not skip_reset:
            self.reset()

    def reset(self):
        # logging.critical('FixedRewardWorld.reset called')
        self.t = 0
        overlap_exit = True

        while overlap_exit:
            self.agent_room = self.np_random.choice(self.possible_start_rooms)
            Lx, Ly = self.room_sizes[self.agent_room]
            self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
            overlap_exit = self.check_reward_overlap(self.agent_room, self.agent_position)

        obs = self.get_observation(self.agent_room, self.agent_position, is_reset=True)
        # print(obs.min(), obs.max(), obs.shape, self.observation_space)
        assert self.observation_space.contains(obs)
        return obs


    def step(self, action):
        assert len(action) == 2
        new_room, rectified_new_pos, rectified_action = self.movement_logic(action)

        # Decouple those now, could be useful later
        end_traj = self.check_reward_overlap(self.agent_room, self.agent_position)

        try:
            reward = self.check_reward_overlap(self.agent_room, self.agent_position).astype(np.float32)
        except AttributeError:
            if self.check_reward_overlap(self.agent_room, self.agent_position):
                reward = 1.
            else:
                reward = 0.

        if not np.all(rectified_action == action):
            reward = -0.05
        if reward == 0.:
            reward = -0.01

        info = {'new_room': deepcopy(new_room), 'new_pos': deepcopy(rectified_new_pos), 'rectified_action': deepcopy(rectified_action)}

        if self.t >= self.epoch_len:
            end_traj = True
        obs = self.get_observation(deepcopy(new_room), deepcopy(rectified_new_pos), action=deepcopy(rectified_action))
        assert self.observation_space.contains(obs)
        return obs, reward, end_traj, info



class GoalBasedWorld(gym.GoalEnv, FixedRewardWorld):
    def __init__(self, scale_actions=False, **kwargs):
        FixedRewardWorld.__init__(self, **kwargs)
        self.observation_space = spaces.Dict({
                'observation': spaces.Box(low = -10 * np.ones(self.observation_shape), high = 10 * np.ones(self.observation_shape), dtype = np.float32),
                'desired_goal': spaces.Box(low = -10 * np.ones(self.observation_shape), high = 10 * np.ones(self.observation_shape), dtype = np.float32),
                'achieved_goal': spaces.Box(low = -10 * np.ones(self.observation_shape), high = 10 * np.ones(self.observation_shape), dtype = np.float32),
        })

        self.scale_actions = scale_actions

        if not kwargs['skip_reset']:
            self.reset()

    def reset(self):
        self.t = 0
        self.reward_room = self.np_random.choice(self.possible_start_rooms)
        Lx, Ly = self.room_sizes[self.reward_room]
        self.reward_position = np.array([self.np_random.uniform(-Lx+self.reward_area_width, Lx-self.reward_area_width), self.np_random.uniform(-Ly+self.reward_area_width, Ly-self.reward_area_width)])

        overlap_exit = True
        while overlap_exit:
            self.agent_room = self.np_random.choice(self.possible_start_rooms)
            Lx, Ly = self.room_sizes[self.agent_room]
            self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
            overlap_exit = self.check_reward_overlap(self.agent_room, self.agent_position)

        obs = self.get_observation(self.agent_room, self.agent_position, is_reset=True)
        goal_obs = self.get_observation(self.reward_room, self.reward_position, is_reset=True)

        self.goal_obs = deepcopy(goal_obs)

        # logging.critical('[reset] notlegacy: t {}, room {}, pos {}, reward_room {}, reward_pos {}, obs.mean {}, goal_obs.mean {}, self.goal_obs.mean {}'.format(self.t, self.agent_room,
        #                 self.agent_position, self.reward_room, self.reward_position, obs.mean(), goal_obs.mean(), self.goal_obs.mean()))

        return {'observation': obs, 'desired_goal': goal_obs, 'achieved_goal': obs}

    def compute_reward(self, achieved_goal, desired_goal, info):
        rewards = []
        for i in info:
            r = 0
            # logging.critical(info)
            room = i['new_room']
            pos = i['new_pos']
            reward_position = i['reward_position']
            reward_room = i['reward_room']
            bumped = i['bumped']
            if bumped:
                r -= 0.05
            overlaps = np.logical_and(room == reward_room, np.max(np.abs(pos-reward_position))<=self.reward_area_width)
            if overlaps:
                r += 1.
            if r == 0.:
                r=-0.01
            rewards.append(r)
        return rewards

    def step(self, action):
        assert len(action) == 2

        if self.scale_actions:
            action = self.scale * action

        new_room, rectified_new_pos, rectified_action = self.movement_logic(action)

        # Decouple those now, could be useful later
        end_traj = self.check_reward_overlap(self.agent_room, self.agent_position)
        reward = self.check_reward_overlap(self.agent_room, self.agent_position).astype(np.float32)

        if not np.all(rectified_action == action):
            reward = -0.05
        if reward == 0.:
            reward = -0.01

        info = {'new_room': deepcopy(new_room), 'new_pos': deepcopy(rectified_new_pos), 'rectified_action': deepcopy(rectified_action),
                'reward_room': deepcopy(self.reward_room), 'reward_position': deepcopy(self.reward_position), 'bumped': deepcopy(not np.all(rectified_action == action))}

        if self.t >= self.epoch_len:
            end_traj = True

        obs = self.get_observation(self.agent_room, self.agent_position, action=rectified_action)
        # goal_obs = self.get_observation(self.reward_room, self.reward_position, is_reset=True)

        complete_obs = {'observation': obs, 'desired_goal': deepcopy(self.goal_obs), 'achieved_goal': obs}


        # logging.critical('[step] notlegacy: t {}, action {}, room {}, pos {}, obs.mean {}, self.goal_obs.mean {}'.format(self.t, action, self.agent_room,
        #                 self.agent_position, obs.mean(), deepcopy(self.goal_obs).mean()))
        # logging.critical(info)
        assert self.observation_space.contains(complete_obs)
        return complete_obs, reward, end_traj, info


# Variants (for-v1, v2, etc...)
class FixedRewardPreprocessedWorld(FixedRewardWorld):
    def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', chosen_reward_pos='Default',
            epoch_len=20, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/', im_availability=.5, corrupt_frac=.5, use_recurrence=True):
        FixedRewardWorld.__init__(self, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, chosen_reward_pos=chosen_reward_pos, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from, use_recurrence=use_recurrence, skip_reset=True)
        self.im_availability = im_availability
        self.corrupt_frac = corrupt_frac
        self.use_recurrence = use_recurrence
        self.reset()

    # Only need to override the get_observation method
    def get_observation(self, room, position, action=None, is_reset=False):
        relative_position = tch.from_numpy(self.positions_blob[room] - position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0)).float()
        color = tch.from_numpy(self.colors_blob[room]).float().to(self.retina.device)
        image_batch = self.retina.activity(relative_position, color)

        with tch.set_grad_enabled(False):
            image_reps = self.preprocessor.get_representation(image_batch)

            if (not self.use_recurrence) or is_reset:
                self.current_state = deepcopy(image_reps)
                return image_reps.squeeze(0).cpu().numpy()

            # don't corrupt if init...
            if self.np_random.uniform() > self.im_availability and not is_reset:
                if self.np_random.uniform() < self.corrupt_frac:
                    image_reps = image_reps[:, self.np_random.permutation(self.state_size)]
                else:
                    image_reps *= 0.

            if action is None:
                raise RuntimeError('Only at reset is it acceptable to have action=None')
            else:
                action_tch = tch.from_numpy(action).unsqueeze(0).to(self.retina.device)
                z_encoding = self.preprocessor.get_z_encoding(action_tch)

            # print(self.current_state.shape, image_reps.shape, z_encoding.shape)
            new_state, _, _, _ = self.preprocessor.update_internal_state(self.current_state, image_reps, z_encoding)
            self.current_state = deepcopy(new_state)

        new_state = new_state.squeeze(0).cpu().numpy()
        # print(new_state.shape, self.observation_space)
        # print(new_state.min(), new_state.max())
        return new_state


class GoalBasedPreprocessedWorld(GoalBasedWorld):
    def __init__(self, scale_actions=False, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', epoch_len=30, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/', im_availability=.5, corrupt_frac=.5, use_recurrence=True):
        self.use_recurrence = use_recurrence
        self.im_availability = im_availability
        self.corrupt_frac = corrupt_frac
        GoalBasedWorld.__init__(self, scale_actions=scale_actions, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from, use_recurrence=use_recurrence, skip_reset=True)
        logging.critical('Done initialiazing GoalBasedWorld')
        self.reset()
        logging.critical('Done initialiazing GoalBasedPreprocesedWorld')

    def get_observation(self, room, position, action=None, is_reset=False):
        relative_position = tch.from_numpy(self.positions_blob[room] - position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0)).float()
        color = tch.from_numpy(self.colors_blob[room]).float().to(self.retina.device)
        image_batch = self.retina.activity(relative_position, color)

        with tch.set_grad_enabled(False):
            image_reps = self.preprocessor.get_representation(image_batch)

            if (not self.use_recurrence) or is_reset:
                self.current_state = deepcopy(image_reps)
                # return image_reps.squeeze(0).cpu().numpy()
                return image_reps.view(self.preprocessor.representation_size).detach().cpu().numpy()


            if self.np_random.uniform() > self.im_availability and not is_reset:
                # logging.critical('entered corruption block')
                if self.np_random.uniform() < self.corrupt_frac:
                    image_reps = image_reps[:, self.np_random.permutation(self.state_size)]
                else:
                    image_reps *= 0.


            if action is None:
                raise RuntimeError('Only at reset is it acceptable to have action=None')
            else:
                action_tch = tch.from_numpy(action).unsqueeze(0).to(self.retina.device)
                z_encoding = self.preprocessor.get_z_encoding(action_tch)

            new_state, _, _, _ = self.preprocessor.update_internal_state(self.current_state, image_reps, z_encoding)
            self.current_state = deepcopy(new_state)

        new_state = new_state.squeeze(0).cpu().numpy()

        return new_state





# Legacy, to try and find the problem...


# class LegacyGoalBasedWorld(gym.GoalEnv):
#     # def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', chosen_reward_pos='TopRight', epoch_len=20, seed=0, load_preprocessor_from=None):
#     def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', epoch_len=20, seed=0, load_preprocessor_from=None, **kwargs):
#         logging.critical('Using Legacy GoalBasedWorld')
#         self.epoch_len = epoch_len
#
#         self.retina = Retina(n=64**2, bounds=[-.5,.5], widths=[.3, .5], device_name='cuda')
#         self.device = self.retina.device
#         self.map_name = map_name
#         self.objects_layout_name = objects_layout_name
#
#         env_params = deepcopy(environment_register[map_name])
#         self.n_rooms = len(env_params['room_centers'])
#         logging.critical('n_rooms in our env : {}'.format(self.n_rooms))
#         self.room_labels = ["Room {}".format(i) for i in range(self.n_rooms)]
#         self.room_centers = env_params['room_centers']
#         self.room_sizes = env_params['room_sizes']
#         self.room_exits = env_params['room_exits']
#         self.room_objects = env_params['possible_layouts'][objects_layout_name]
#         self.scale = scale
#
#         self.max_objects_per_room = np.max([objs_dict['positions'].shape[0] for objs_dict in self.room_objects])
#         self.colors_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 3))
#         self.positions_blob = np.zeros((self.n_rooms, self.max_objects_per_room, 2))
#         for room_idx in range(self.n_rooms):
#             for obj_idx in range(self.room_objects[room_idx]['colors'].shape[0]):
#                 self.colors_blob[room_idx, obj_idx] = self.room_objects[room_idx]['colors'][obj_idx]
#                 self.positions_blob[room_idx, obj_idx] = self.room_objects[room_idx]['positions'][obj_idx]
#
#         self.room_sizes = scale * self.room_sizes
#         self.room_centers = scale * self.room_centers
#
#         for i in range(self.n_rooms):
#             self.room_objects[i]['positions'] = scale * self.room_objects[i]['positions']
#             for obj_idx in range(len(self.room_exits[i])):
#                 self.room_exits[i][obj_idx]['goes_to'][1] = scale * np.array(self.room_exits[i][obj_idx]['goes_to'][1])
#                 self.room_exits[i][obj_idx]['x'] = scale * self.room_exits[i][obj_idx]['x']
#                 self.room_exits[i][obj_idx]['y'] = scale * self.room_exits[i][obj_idx]['y']
#
#         self.reward_area_width = reward_area_width * self.scale
#
#         # Useful for losses; for now, only rooms with no objects
#         self.rooms_not_to_start_in = [i for i in range(self.n_rooms) if np.all(self.room_objects[i]['colors'] == 0.)]
#         self.possible_start_rooms = [i for i in range(self.n_rooms) if not np.all(self.room_objects[i]['colors'] == 0.)]
#
#         if self.map_name == 'DoubleDonut' and self.objects_layout_name == 'Ambiguous':
#              self.possible_start_rooms = np.array([0,1,2,3,4,5,6,8,9,10,12,13,14,15]) # Do not allow start from ambiguous rooms
#
#         logging.critical('Allowed starting rooms : {}'.format(self.possible_start_rooms))
#         self.set_seed(seed)
#
#         self.seed_value = seed
#
#         if load_preprocessor_from is not None:
#             with open(load_preprocessor_from+'/full_params.json', mode='r') as f:
#                 params = json.load(f)['net_params']['options']
#             self.preprocessor = BigResetNetwork(**params).to(self.device)
#             logging.critical('Attempting to load preprocessor from : {}'.format(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value)))
#             self.preprocessor.load(load_preprocessor_from+'seed{}/best_net.tch'.format(self.seed_value))
#         else:
#             self.preprocessor = None
#
#         # Gym specific stuff
#         self.action_space = gym.spaces.Box(low=.8*scale *np.array([-1.0, -1.0]), high=.8*scale *np.array([1.0, 1.0]), dtype=np.float32) # Continuous actions, bounded for simplicity
#         if load_preprocessor_from is None:
#             self.observation_shape = (64**2, 3)
#         else:
#             self.observation_shape = (self.preprocessor.representation_size,)
#
#         self.observation_space = spaces.Dict({
#                 'observation': spaces.Box(low = -100 * np.ones(self.observation_shape), high = 100 * np.ones(self.observation_shape), dtype = np.float32),
#                 'desired_goal': spaces.Box(low = -100 * np.ones(self.observation_shape), high = 100 * np.ones(self.observation_shape), dtype = np.float32),
#                 'achieved_goal': spaces.Box(low = -100 * np.ones(self.observation_shape), high = 100 * np.ones(self.observation_shape), dtype = np.float32),
#         })
#         self.reset()
#         logging.critical('Done initializing LagacyGoalBasedWorld')
#
#     def set_seed(self, seed=None):
#         logging.critical('LegacyGoalBasedWorld.seed called')
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#
#     def render_background(self, ax_to_use=None):
#         if ax_to_use is None:
#             fig, ax = plt.subplots()
#         else:
#             ax = ax_to_use
#
#         ax.set_facecolor([0,0,0,0])
#         ax.set_axis_off()
#         ax.set_aspect('equal')
#
#         ax.set_xlim([np.min(self.room_centers[:,0])-self.scale-.05, np.max(self.room_centers[:,0])+self.scale+.05])
#         ax.set_ylim([np.min(self.room_centers[:,1])-self.scale-.05, np.max(self.room_centers[:,1])+self.scale+.05])
#
#         # This one is common to all environments, it determines the enclosing area
#         rect = patches.Rectangle((np.min(self.room_centers[:,0])-self.scale, np.min(self.room_centers[:,1]-self.scale)),
#                                 np.max(self.room_centers[:,0]) - np.min(self.room_centers[:,0])+ 2*self.scale,
#                                 np.max(self.room_centers[:,1]) - np.min(self.room_centers[:,1])+ 2*self.scale,
#                                 linewidth=1, edgecolor='k',facecolor=[0,0,0,0])
#         ax.add_patch(rect)
#
#         if self.map_name == 'SnakePath':
#             ax.plot([-3 *self.scale, self.scale], [-self.scale, -self.scale], c='k', linewidth=3)
#             ax.plot([-self.scale, 3*self.scale], [self.scale, self.scale], c='k', linewidth=3)
#             ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
#             ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
#             ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#             ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#         elif self.map_name == 'DonutPath':
#             rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
#             ax.add_patch(rect)
#             ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
#             ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
#             ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#             ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#         elif self.map_name == 'DoubleDonut':
#             rect = patches.Rectangle((-self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
#             ax.add_patch(rect)
#
#             ax.plot([-3*self.scale, 3*self.scale], [-self.scale, -self.scale], c='k', ls='--')
#             ax.plot([-3*self.scale, 3*self.scale], [self.scale, self.scale], c='k', ls='--')
#             ax.plot([-self.scale, -self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#             ax.plot([self.scale, self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#
#             rect = patches.Rectangle((5*self.scale, -self.scale), 2* self.scale, 2*self.scale, linewidth=1, edgecolor='k',facecolor=[0,0,0,.5])
#             ax.add_patch(rect)
#
#             ax.plot([3*self.scale, 3*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#             ax.plot([3*self.scale, 9*self.scale], [-self.scale, -self.scale], c='k', ls='--')
#             ax.plot([3*self.scale, 9*self.scale], [self.scale, self.scale], c='k', ls='--')
#             ax.plot([5*self.scale, 5*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#             ax.plot([7*self.scale, 7*self.scale], [-3*self.scale, 3*self.scale], c='k', ls='--')
#
#         if ax_to_use is None:
#             return fig, ax
#         else:
#             return ax
#
#
#
#     def get_observation(self, room, position):
#         # position = tch.from_numpy(self.positions_blob[room] + position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0)).float()
#         all_positions =  tch.from_numpy((self.positions_blob[room] - position[np.newaxis, :].repeat(self.max_objects_per_room, axis=0))).float().to(self.retina.device)
#
#         # all_positions = all_positions.to(self.retina.device)
#         color = tch.from_numpy(self.colors_blob[room]).float().to(self.retina.device)
#         # all_positions = all_positions.reshape((1, *all_positions.shape))
#         # color = color.reshape((1, *color.shape))
#         # print(all_positions.shape, color.shape)
#         # all_colors = self.colors_blob[rooms].astype(np.float32)
#         image_batch = self.retina.activity(all_positions, color)
#
#         if self.preprocessor is not None:
#             with tch.set_grad_enabled(False):
#                 image_batch = self.preprocessor.get_representation(image_batch)
#             image_batch = image_batch.view(self.preprocessor.representation_size).detach().cpu().numpy()
#         else:
#             # logging.critical(image_batch.shape)
#             image_batch = image_batch.squeeze(0).detach().cpu().numpy()
#
#         return image_batch
#
#     def __check_reward_overlap(self, room, pos):
#         overlaps = np.logical_and(room == self.reward_room, np.max(np.abs(pos-self.reward_position))<=self.reward_area_width)
#         # if overlaps:
#         #     logging.critical('reward overlap event triggered')
#         return overlaps
#
#     # Some engineering to be done here
#     def compute_reward(self, achieved_goal, desired_goal, info):
#         rewards = []
#         for i in info:
#             r = 0
#             # logging.critical(info)
#             room = i['new_room']
#             pos = i['new_pos']
#             reward_position = i['reward_position']
#             reward_room = i['reward_room']
#             bumped = i['bumped']
#             if bumped:
#                 r -= 0.05
#             overlaps = np.logical_and(room == reward_room, np.max(np.abs(pos-reward_position))<=self.reward_area_width)
#             if overlaps:
#                 r += 1.
#             if r == 0.:
#                 r=-0.01
#             rewards.append(r)
#         return rewards
#
#     def reset(self):
#         # logging.critical('reset called')
#         self.t = 0
#
#         self.reward_room = self.np_random.choice(self.possible_start_rooms)
#         Lx, Ly = self.room_sizes[self.reward_room]
#         self.reward_position = np.array([self.np_random.uniform(-Lx+self.reward_area_width, Lx-self.reward_area_width), self.np_random.uniform(-Ly+self.reward_area_width, Ly-self.reward_area_width)])
#
#
#         overlap_exit = True
#         while overlap_exit:
#             self.agent_room = self.np_random.choice(self.possible_start_rooms)
#             Lx, Ly = self.room_sizes[self.agent_room]
#             self.agent_position = np.array([self.np_random.uniform(-Lx, Lx), self.np_random.uniform(-Ly, Ly)])
#             overlap_exit = self.__check_reward_overlap(self.agent_room, self.agent_position)
#
#         obs = self.get_observation(self.agent_room, self.agent_position)
#         goal_obs = self.get_observation(self.reward_room, self.reward_position)
#
#         self.goal_obs = deepcopy(goal_obs)
#
#         # logging.critical('[reset] legacy: t {}, room {}, pos {}, reward_room {}, reward_pos {}, obs.mean {}, goal_obs.mean {}, self.goal_obs.mean {}'.format(self.t, self.agent_room,
#         #                 self.agent_position, self.reward_room, self.reward_position, obs.mean(), goal_obs.mean(), self.goal_obs.mean()))
#
#         return {'observation': obs, 'desired_goal': goal_obs, 'achieved_goal': obs}
#
#
#     def step(self, action):
#         assert len(action) == 2
#         action_bkp = deepcopy(action)
#         bkp_x, bkp_y = deepcopy(self.agent_position)
#         room_bkp = deepcopy(self.agent_room)
#
#         new_pos = np.array([self.agent_position[0] + action[0], self.agent_position[1] + action[1]])
#         Lx, Ly = self.room_sizes[self.agent_room]
#
#         invalid_x = new_pos[0]>Lx or new_pos[0]<-Lx
#         invalid_y = new_pos[1]>Ly or new_pos[1]<-Ly
#
#         if invalid_x and invalid_y:
#             if self.np_random.uniform() < .5:
#                 action[0] -= new_pos[0] - np.clip(new_pos[0], -Lx, Lx)
#                 new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
#                 invalid_x = False
#             else:
#                 action[1] -= new_pos[1] - np.clip(new_pos[1], -Ly, Ly)
#                 new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
#                 invalid_y = False
#
#
#         changed_room = False
#         if invalid_y:
#             for exit in self.room_exits[self.agent_room]:
#                 if changed_room:
#                     break
#                 if exit['axis'] =='horizontal':
#                     if np.clip(new_pos[1], -Ly, Ly) == exit['y']:
#                         logging.debug('crossed horizontal door')
#                         used_exit = deepcopy(exit)
#                         changed_room = True
#
#         if invalid_x:
#             for exit in self.room_exits[self.agent_room]:
#                 if changed_room:
#                     break
#                 if exit['axis'] =='vertical':
#                     if np.clip(new_pos[0], -Lx, Lx) == exit['x']:
#                         logging.debug('crossed vertical door')
#                         used_exit = deepcopy(exit)
#                         changed_room = True
#
#         if not changed_room:
#             new_room = room_bkp
#         else:
#             new_room = used_exit['goes_to'][0]
#             new_pos = new_pos + self.room_centers[room_bkp, :2] - self.room_centers[new_room, :2]
#
#         rectified_new_pos = np.zeros(2)
#         rectified_new_pos[0] = np.clip(new_pos[0], -Lx, Lx)
#         rectified_new_pos[1] = np.clip(new_pos[1], -Ly, Ly)
#
#         rectified_action = action + rectified_new_pos - new_pos
#
#         self.agent_room = deepcopy(new_room)
#         self.agent_position = deepcopy(rectified_new_pos)
#
#         self.t += 1
#
#         # Decouple those now, could be useful later
#         end_traj = self.__check_reward_overlap(self.agent_room, self.agent_position)
#         reward = self.__check_reward_overlap(self.agent_room, self.agent_position).astype(np.float32)
#         if not np.all(rectified_action == action):
#             reward = -0.05
#         if reward == 0.:
#             reward = -0.01
#         info = {'new_room': deepcopy(new_room), 'new_pos': deepcopy(rectified_new_pos), 'rectified_action': deepcopy(rectified_action),
#                 'reward_room': deepcopy(self.reward_room), 'reward_position': deepcopy(self.reward_position), 'bumped': deepcopy(not np.all(rectified_action == action))}
#
#         if self.t >= self.epoch_len:
#             end_traj = True
#         obs = self.get_observation(self.agent_room, self.agent_position)
#
#         # logging.critical('[step] legacy: t {}, action {} room {}, pos {}, obs.mean {}, goal_obs.mean {}, self.goal_obs.mean {}'.format(self.t, action, self.agent_room,
#         #                 self.agent_position, obs.mean(), deepcopy(self.goal_obs).mean(), self.goal_obs.mean()))
#         # logging.critical(info)
#
#
#         return {'observation': obs, 'desired_goal': deepcopy(self.goal_obs), 'achieved_goal': obs}, reward, end_traj, info
#
#
#
#     # Below that are functions used for legacy interface
#     def set_agent_position(self, room, xy=(0,0)):
#         self.agent_room = room
#         Lx, Ly = self.room_sizes[self.agent_room]
#         invalid_x = xy[0]>Lx or xy[0]<-Lx
#         invalid_y = xy[1]>Ly or xy[1]<-Ly
#         if invalid_x or invalid_y:
#             raise RuntimeError('Invalid xy initialization for current room')
#         self.agent_position = np.array([*xy])
#
#     def set_reward_position(self, room, xy=(0,0)):
#         self.reward_room = room
#         self.reward_position = np.array([*xy])
#
#     def render_template(self, ax_to_use=None):
#         # print('just inside render template', self.reward_room, self.reward_position)
#         if ax_to_use is None:
#             fig, ax = self.render_background(ax_to_use=None)
#         else:
#             ax = self.render_background(ax_to_use=ax_to_use)
#
#         for room in range(self.n_rooms):
#             for xy, col in zip(self.room_objects[room]['positions'], self.room_objects[room]['colors']):
#                 xy0 = self.room_centers[room, :2]
#                 rect = patches.Rectangle(xy+xy0-.05*self.scale, .2*self.scale, .2*self.scale, linewidth=1, edgecolor='k', facecolor=col)
#                 ax.add_patch(rect)
#
#         print(self.reward_room, self.reward_position)
#         exit_rect = patches.Rectangle(self.room_centers[self.reward_room, :2]+self.reward_position-self.reward_area_width, 2*self.reward_area_width, 2*self.reward_area_width, facecolor='gray', zorder=100, alpha=.5)
#         ax.add_patch(exit_rect)
#
#         if ax_to_use is None:
#             return fig, ax
#         else:
#             return ax
#
#     def get_images(self, rooms, positions):
#         desired_shape = deepcopy(rooms.shape)
#         rooms = rooms.flatten().astype(int)
#         positions = positions.reshape(-1, 2)
#         # logging.critical('positions info : shape {}, type {}'.format(positions.shape, positions.dtype))
#         # all_positions = tch.from_numpy(self.positions_blob[rooms] + positions[:, np.newaxis, :].repeat(self.max_objects_per_room, axis=1)).float()
#         all_positions = tch.from_numpy(self.positions_blob[rooms] - positions[:, np.newaxis, :].repeat(self.max_objects_per_room, axis=1)).float()
#
#         all_positions = all_positions.to(self.retina.device)
#         all_colors = tch.from_numpy(self.colors_blob[rooms]).float().to(self.retina.device)
#         image_batch = self.retina.activity(all_positions, all_colors)
#
#         return image_batch.reshape(*desired_shape, self.retina.n, 3)#.cpu().numpy()
#
#
#     def __replay_one_traj(self, actions, start_room=None, start_pos=None, reward_room=None, reward_pos=None):
#         self.reset()
#         epoch_len = actions.shape[0]
#         positions = np.zeros((epoch_len+1,2))
#         validated_actions = np.zeros((epoch_len,2))
#         rooms = np.zeros((epoch_len+1))
#
#         if start_room is None:
#             room, pos = self.agent_room, self.agent_position
#             reward_room, reward_pos = self.reward_room, self.reward_position
#         else:
#             room, pos = start_room, start_pos
#             self.reward_room = reward_room
#             self.reward_position = reward_pos
#             room = int(room)
#             self.set_agent_position(room, (pos[0], pos[1]))
#
#         # logging.critical('in replay one traj {}'.format([self.reward_room, self.reward_position, self.room_centers[self.reward_room, :2], ]))
#         positions[0] = pos
#         rooms[0] = room
#
#         for idx, action in enumerate(actions):
#             # new_room, new_pos, rectified_action, reward, is_terminal = self.step(action)
#             obs, reward, end_traj, info = self.step(action)
#             new_room, new_pos, rectified_action = info['new_room'], info['new_pos'], info['rectified_action']
#             logging.debug('Start in room {} at ({},{}) and end in room {} at ({},{}) with tot_dep ({},{})'.format(room, *pos, new_room, *new_pos, *rectified_action))
#             validated_actions[idx] = rectified_action
#             positions[idx+1] = new_pos
#             rooms[idx+1] = new_room
#             pos = new_pos
#             room = new_room
#
#             if end_traj:
#                 positions[idx+1:] = new_pos
#                 rooms[idx+1:] = new_room
#                 validated_actions[idx+1:] = 0.
#                 break
#
#
#         return rooms, positions, validated_actions
#
#
#     def static_replay(self, actions_batch, start_rooms=None, start_pos=None):
#         actions_batch_local = deepcopy(actions_batch)
#         batch_size = actions_batch.shape[0]
#         epoch_len = actions_batch.shape[1]
#
#         if start_rooms is not None:
#             assert start_pos is not None
#             assert start_rooms.shape[0] == start_pos.shape[0]
#             assert start_pos.shape[1] == 2
#
#         rooms = np.zeros((batch_size, epoch_len+1))
#         positions = np.zeros((batch_size, epoch_len + 1, 2))
#         validated_actions = np.zeros((batch_size, epoch_len, 2))
#         rewards = np.zeros((batch_size, epoch_len+1))
#         irrelevant_times = np.zeros((batch_size, epoch_len+1))
#
#         # NOTE: making this multi-threaded seems smart, but at least in my tests its either slow, buggy, or both
#         for b in range(batch_size):
#             if start_rooms is None:
#                 room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
#             else:
#                 room, pos, act, =  self.__replay_one_traj(actions_batch_local[b], start_room=start_rooms[b], start_pos=start_pos[b])
#
#             rooms[b, :] = room
#             positions[b, :] = pos
#             validated_actions[b, :] = act
#
#         logging.debug('Done with static_replay in environments.py')
#         return rooms, positions, validated_actions
#
#
#     @staticmethod
#     def __add_arrows(line, size=15, color=None, zorder=-1):
#
#         if color is None:
#             color = line.get_color()
#
#         xdata = line.get_xdata()
#         ydata = line.get_ydata()
#
#         x_ends = .5* (xdata[:-1] + xdata[1:])
#         y_ends = .5* (ydata[:-1] + ydata[1:])
#
#         for x_start, x_end, y_start, y_end in zip(xdata, x_ends, ydata, y_ends):
#             line.axes.annotate('',
#                 xytext=(x_start, y_start),
#                 xy=(x_end, y_end),
#                 arrowprops=dict(arrowstyle="->", color=color),
#                 size=size, zorder=-1
#             )
#
#
#     def plot_trajectory(self, actions, start_room=None, start_pos=None, reward_room=None, reward_pos=None, ax_to_use=None, save_file=None, color=None, marker='+', zorder=500, show_lines=False, show_arrows=False, s=32, return_fig=False, **kwargs):
#         # By default, color is set to show time, but can be overridden
#         if color is None:
#             time_based_norm = matplotlib.colors.Normalize(vmin=0, vmax=actions.shape[0]+1)
#             cmap = plt.get_cmap('jet')
#             color = cmap(time_based_norm(range(actions.shape[0]+1)))
#
#         if ax_to_use is None:
#             if self.map_name != 'DoubleDonut':
#                 fig, ax = plt.subplots(figsize=(5,5))
#             elif self.map_name == 'DoubleDonut':
#                 fig, ax = plt.subplots(figsize=(10,5))
#
#         else:
#             ax = ax_to_use
#
#         if start_room is None:
#             room, pos, act =  self.__replay_one_traj(actions, start_room=None, start_pos=None, reward_room=None, reward_pos=None)
#         else:
#             # logging.critical([reward_room, reward_pos])
#             room, pos, act =  self.__replay_one_traj(actions, start_room=start_room, start_pos=start_pos, reward_room=reward_room, reward_pos=reward_pos)
#
#         ax = self.render_template(ax_to_use=ax)
#         absolute_pos = pos+self.room_centers[room.astype(int), :2]
#
#         ax.scatter(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, marker=marker, s=s)
#
#         if show_lines:
#             lines = ax.plot(absolute_pos[:,0], absolute_pos[:,1], c=color, zorder=zorder, ls='-')
#             if show_arrows:
#                 self.__add_arrows(lines[0], zorder=zorder, color=color)
#
#         if save_file is not None:
#             os.makedirs('/'.join(save_file.split('/')[:-1]), exist_ok=True)
#             fig.savefig(save_file)
#             plt.close(fig)
#         elif return_fig:
#             return fig
#         else:
#             return ax
#
# # # Variants (for-v1, v2, etc...)
# # class FixedRewardPreprocessedWorld(FixedRewardWorld):
# #     def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', chosen_reward_pos='Default', epoch_len=20, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/'):
# #         FixedRewardWorld.__init__(self, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, chosen_reward_pos=chosen_reward_pos, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from)
#
# class LegacyGoalBasedPreprocessedWorld(LegacyGoalBasedWorld):
#     def __init__(self, map_name='SnakePath', scale=.5, reward_area_width = .3, objects_layout_name='Default', epoch_len=30, seed=0, load_preprocessor_from='out/SnakePath_Default/end_to_end/default/', **kwargs):
#         LegacyGoalBasedWorld.__init__(self, map_name=map_name, scale=scale, reward_area_width = reward_area_width, objects_layout_name=objects_layout_name, epoch_len=epoch_len, seed=seed, load_preprocessor_from=load_preprocessor_from)
#
# #
# #
# #
# #
#



if __name__ == '__main__':
    os.makedirs('out/env_render_templates/', exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    map_name = 'SnakePath'
    objects_layout_name = 'Default'

    # env = LegacyWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos='plo')
    # env = FixedRewardWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos='Default')
    #
    # raise RuntimeError

    for chosen_reward_pos in ['None', 'Default', 'TopRight']:
        env = FixedRewardWorld(map_name=map_name, objects_layout_name=objects_layout_name, scale=.5, chosen_reward_pos=chosen_reward_pos)
        # env = World(map_name=map_name, objects_layout_name=objects_layout_name)
        fig, ax = env.render_template()
        fig.savefig('out/env_render_templates/{}_{}_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos))
        plt.close('all')

        actions_batch = env.scale * np.random.randn(10, 20, 2)
        start_rooms = np.random.choice(env.n_rooms, size=(100,))
        start_pos = np.random.uniform(-env.scale, env.scale, size=(100, 2))
        bkp_actions = deepcopy(actions_batch)
        _, _, validated_actions = env.static_replay(actions_batch, start_rooms=start_rooms, start_pos=start_pos)

        assert np.all(bkp_actions == actions_batch), "Side-effect detected in static replay"
        assert not np.all(validated_actions == bkp_actions), "Static replay made no change to any of the trajectories, this is suspicious"

        for b in range(10):
            env.plot_trajectory(validated_actions[b], start_room=start_rooms[b], start_pos=start_pos[b], save_file='out/env_render_templates/{}_{}_{}/trajectories/traj_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos, b))

        if map_name in ['DoubleDonut', 'SnakePath']:
            deliberate_actions = np.reshape(meaningful_trajectories[map_name] * env.scale, (1,)+meaningful_trajectories[map_name].shape)
            print(deliberate_actions)
            rooms, positions, deliberate_actions = env.static_replay(deliberate_actions, start_rooms=np.zeros(5, dtype=int), start_pos=np.zeros((5, 2)))
            print(deliberate_actions)

            deliberate_actions = deliberate_actions[0]
            rooms = rooms[0].astype(int)
            positions = positions[0]

            fig, ax = plt.subplots(figsize=(10,5))
            ax = env.plot_trajectory(deliberate_actions, start_room=0, start_pos=np.array([0.,0.]), ax_to_use=ax, marker = 'x', zorder=-5)
            fig.savefig('out/env_render_templates/{}_{}_{}/trajectories/deliberate_traj.pdf'.format(map_name, objects_layout_name, chosen_reward_pos))

            for b in range(5):
                perturbed_positions = np.clip(positions+.2*env.scale*np.random.uniform(-1, 1, size=positions.shape), -env.scale, env.scale)
                global_positions = perturbed_positions + env.room_centers[rooms, :2]
                perturbed_actions = global_positions[1:] - global_positions[:-1]

                fig, ax = plt.subplots(figsize=(10,5))
                # env.plot_trajectory(deliberate_actions + .02*env.scale*np.mean(np.abs(deliberate_actions))*np.random.randn(*deliberate_actions.shape), ax_to_use=ax, start_room=0, start_pos=np.array([0.,0.]))
                env.plot_trajectory(perturbed_actions, ax_to_use=ax, start_room=0, start_pos=np.array([0.,0.]))
                fig.savefig('out/env_render_templates/{}_{}_{}/trajectories/deliberate_traj_noisy_{}.pdf'.format(map_name, objects_layout_name, chosen_reward_pos, b))
