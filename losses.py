'''
This file contains the script to compute relevant losses from a net and an env
'''
from torch.nn.functional import mse_loss, l1_loss, nll_loss, log_softmax
from copy import deepcopy
import torch as tch
import logging
import numpy as np
import time

class TransitionLoss:
    def __init__(self, policy, batch_size=256, use_full_space=False):
        self.policy = policy
        self.batch_size = batch_size

        # If False (default) use only transitions that can be done with one action
        # else, use any two images from the environment (mostly relevant to backwards)
        self.use_full_space = use_full_space

    def __call__(self, net, env):
        if not self.use_full_space:
            actions = self.policy.get_batch_of_actions(batch_size=self.batch_size, epoch_len=1)
            rooms, positions, actions = env.static_replay(actions)
        else:
            rooms = np.random.choice(env.possible_start_rooms, size=(self.batch_size, 2))
            x_room = np.random.uniform(-env.scale, env.scale, size=(self.batch_size, 2))
            y_room = np.random.uniform(-env.scale, env.scale, size=(self.batch_size, 2))
            positions = np.stack([x_room, y_room], axis=2)
            x_global = x_room + env.room_centers[rooms][:,:,0]
            y_global = y_room + env.room_centers[rooms][:,:,1]
            global_positions = np.stack([x_global, y_global], axis=2)
            actions = tch.from_numpy(global_positions[:, 1, :] - global_positions[:, 0, :]).unsqueeze(1).numpy()

        images = env.get_images(rooms, positions) #retinal states, (bs, T+1=2, ret_res**2, 3)
        images_intact = deepcopy(images)

        reps0, reps1 = net.get_representation(images[:, 0]), net.get_representation(images[:, 1])
        actions_hat = net.backward_model(reps0, reps1)

        action_encodings = net.get_z_encoding(tch.from_numpy(actions[:,0]).float().to(net.device))
        reps_hat = net.forward_model(reps0, action_encodings)
        reinferred_actions = net.backward_model(reps0, reps_hat)

        loss_bkw = mse_loss(actions_hat, tch.from_numpy(actions[:,0]).float().to(net.device))
        loss_fwd = mse_loss(reps_hat, reps1)
        loss_reinference = mse_loss(reinferred_actions, tch.from_numpy(actions[:,0]).float().to(net.device))

        return loss_fwd, loss_bkw, loss_reinference

class PathIntegrationLoss:
    def __init__(self, policy, batch_size=64, epoch_len=20, im_availability=.15, corruption_rate=.5, additive_noise=0.):
        self.policy = policy
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.im_availability = im_availability
        self.corruption_rate = corruption_rate
        self.additive_noise = additive_noise # NOTE: this is an additive noise; if a list, then we randomly choose one
        logging.critical('Using path integration loss with, image_availability {}, corruption rate {}, additive_noise {} [seed{}]'.format(self.im_availability,
                                                                                                                            self.corruption_rate, self.additive_noise, self.policy.seed))

    def __call__(self, net, env):
        tic = time.time()
        actions = self.policy.get_batch_of_actions(batch_size=self.batch_size, epoch_len=self.epoch_len)
        rooms, positions, actions = env.static_replay(actions)

        # Note: slightly dirty, but works fine as long as you don't put anything too exotic as additive_noise
        try:
            if len(self.additive_noise) > 1:
                this_step_noise = np.random.choice(self.additive_noise)
        except:
            this_step_noise = self.additive_noise

        zs = actions + this_step_noise * self.policy.rng.randn(*actions.shape)

        cumulated_actions = tch.cumsum(tch.from_numpy(actions), dim=1).to(net.device).float()
        images = env.get_images(rooms, positions) #retinal states, (bs, T+1=2, ret_res**2, 3)
        representations = net.get_representation(images.view(self.batch_size * (self.epoch_len+1), -1, 3)).view(self.batch_size, (self.epoch_len+1), -1)
        null_images_rep =  net.get_representation(0. * images.view(self.batch_size * (self.epoch_len+1), -1, 3)).view(self.batch_size, (self.epoch_len+1), -1)
        z_encodings =  net.get_z_encoding(tch.from_numpy(zs).view(self.batch_size * (self.epoch_len), 2).float().to(net.device)).view(self.batch_size, (self.epoch_len), -1)


        ims_to_perturb =  tch.bernoulli((1.-self.im_availability) * tch.ones(self.batch_size, self.epoch_len+1))
        # Never mess with anchor point, otherwise whole trajectory is meaningless
        ims_to_perturb[:, 0] = tch.zeros_like(ims_to_perturb[:, 0])

        # For images that should be perturbed, choose between "corruption" and "drop"
        if self.corruption_rate > 0:
            corrupt = tch.where(tch.bernoulli(self.corruption_rate * tch.ones(self.batch_size, self.epoch_len+1)).byte(), ims_to_perturb, tch.zeros(self.batch_size, self.epoch_len+1)).bool()
            drop = tch.logical_and(ims_to_perturb, tch.logical_not(corrupt))
            real_drop = tch.logical_and(drop, tch.bernoulli(0.5 * tch.ones_like(drop)))
            null_image = tch.logical_and(drop, tch.logical_not(real_drop) )
            representations[null_image] = null_images_rep[null_image]
            mask = (1.-drop.float()).unsqueeze(-1).repeat(1, 1, net.representation_size).float().to(net.device)
            representations = mask * representations

            tmp = representations[corrupt]
            # logging.critical(tmp.shape)
            # tmp = tmp[:, tch.randperm(tmp.shape[1])]
            toc = time.time()
            for t in range(tmp.shape[0]):
                tmp[t] = tmp[t, tch.randperm(tmp.shape[1])]

            representations[corrupt] = tmp
        else:
            representations[ims_to_perturb.long()] = null_images_rep[ims_to_perturb.long()]


        outputs, log_gatings, _, out_forward, out_visual = net.do_path_integration(representations, z_encodings, return_all=True)
        gatings = tch.exp(log_gatings)

        # Return all three mainly for diagnostics
        loss_path_integration = mse_loss(outputs, cumulated_actions).float()
        loss_gating = tch.mean(gatings[:, :, 1]).float()
        loss_uncertainty = tch.mean(gatings[:, :, 1]*(1.-gatings[:, :, 1])).float()

        # logging.critical('Total time in PI loss : {}'.format(time.time()-tic))

        return loss_path_integration, loss_gating, loss_uncertainty
