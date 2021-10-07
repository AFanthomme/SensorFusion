'''
In launcher, everything is markup; here, we transform that into objects that interact with each other
'''

import logging
import numpy as np
from torch.optim import SGD, Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
from copy import deepcopy
import torch as tch
import os
from numpy.random import RandomState


from src.networks import network_register
from src.environment import environment_register, FixedRewardWorld
from src.tests import tests_register
from src.losses import *
from src.policy import policy_register
from src.schedules import *

class EndToEndTrainer:
    def __init__(self, params, seed):
        logging.critical(params)
        net_pars = params['net_params']
        env_pars = params['env_params']
        train_pars = params['train_params']
        policy_pars = params['policy_params']
        load_from = params['load_from']

        optimizer_params_scheduler_name = train_pars['optimizer_params_scheduler_name']
        optimizer_params_scheduler_options =  train_pars['optimizer_params_scheduler_options']
        losses_weights_scheduler_name = train_pars['losses_weights_scheduler_name']
        losses_weights_scheduler_options =  train_pars['losses_weights_scheduler_options']

        self.test_suite = params['test_suite']
        self.optimizer_name = train_pars['optimizer_name']

        if self.optimizer_name == 'Adam':
            self.optimizer = Adam
        elif self.optimizer_name == 'SGD':
            self.optimizer = SGD

        self.do_resets = train_pars['do_resets']
        self.n_epochs = train_pars['n_epochs']
        # self.batch_size = train_pars['batch_size']

        self.seed = seed
        self.policy = policy_register[policy_pars['type']](seed=seed, **policy_pars)

        self.net_name = net_pars['net_name']
        self.net_options = net_pars['options']
        self.network = network_register[self.net_name](**self.net_options) # NOTE: useful to define it here so that we can do the initialization and forget about load_rom, but not that relevant

        self.lr_scheduler = lr_scheduler_list[optimizer_params_scheduler_name](**optimizer_params_scheduler_options)
        self.weights_scheduler = weights_scheduler_list[losses_weights_scheduler_name](**losses_weights_scheduler_options)

        self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(-1)
        self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(-1)

        if load_from is not None:
            self.network.load_state_dict(tch.load(load_from+'seed{}/best_net.tch'.format(seed)), strict=False)

        logging.critical(env_pars)
        self.env = FixedRewardWorld(**env_pars, seed=seed)
        self.tuple_loss = TransitionLoss(self.policy, **train_pars['tuple_loss_options'])
        self.pi_loss = PathIntegrationLoss(self.policy, **train_pars['pi_loss_options'])

        self.validation_loss = PathIntegrationLoss(self.policy,
                                                    batch_size=train_pars['pi_loss_options']['batch_size'],
                                                    epoch_len=train_pars['pi_loss_options']['epoch_len'],
                                                    im_availability=train_pars['pi_loss_options']['im_availability'],
                                                    corruption_rate=train_pars['pi_loss_options']['corruption_rate'],
                                                    additive_noise=0.)

        logging.info('Done initializing [seed{}]'.format(seed))

    def do_tests(self, epoch, net, env):
        if (epoch == 0) or (self.test_suite is None):
            return
        for test_name, test_params in self.test_suite.items():
            try:
                if epoch == 'final':
                    raise TypeError
                should_launch = (epoch % test_params['period'] == 0)
            except TypeError:
                should_launch = True

            if should_launch:
                logging.info('Launching test {}'.format(test_name))
                try:
                    os.makedirs(net.save_folder + '{}/'.format(epoch), exist_ok=True)
                except FileExistsError:
                    pass
                tests_register[test_name](net, env, test_params, epoch)
                logging.info('Done running test {}'.format(test_name))

    def train(self):
        logging.info('Starting training [seed{}]'.format(self.seed))
        net = self.network
        env = self.env
        logging.critical(net)

        opt = self.optimizer([
                                {'params': net.gating_module.parameters(), 'lr': self.learning_rates[0]},
                                {'params': net.representation_module.parameters(), 'lr': self.learning_rates[1]},
                                {'params': net.z_encoder_module.parameters(), 'lr': self.learning_rates[2]},
                                {'params': net.backward_module.parameters(), 'lr': self.learning_rates[3]},
                                {'params': net.forward_module.parameters(), 'lr': self.learning_rates[4]},
                            ])

        losses = np.zeros((self.n_epochs+1,6))

        best_net = deepcopy(net)
        best_loss = 1e10

        valid_losses = np.zeros(10)
        for epoch in range(self.n_epochs+1):
            # Compute the losses
            fwd_loss, bkw_loss, reinf_loss = self.tuple_loss(net, env)
            pi_loss, gating_loss, uncertainty_loss = self.pi_loss(net, env)

            if epoch % 10 == 0:
                with tch.set_grad_enabled(False):
                    valid_loss, _, _ = self.validation_loss(net, env)
                    valid_losses[(epoch%100)//10] = valid_loss.item()

            opt.zero_grad()
            tot_loss = self.losses_weights[0] * bkw_loss + self.losses_weights[1] * fwd_loss + self.losses_weights[2] *reinf_loss
            tot_loss = tot_loss +  self.losses_weights[3] * pi_loss  + self.losses_weights[4] * gating_loss + self.losses_weights[5] * uncertainty_loss
            tot_loss.backward()
            opt.step()

            self.do_tests(epoch, net, env)
            losses[epoch, 0] = fwd_loss.item()
            losses[epoch, 1] = bkw_loss.item()
            losses[epoch, 2] = reinf_loss.item()
            losses[epoch, 3] = pi_loss.item()
            losses[epoch, 4] = gating_loss.item()
            losses[epoch, 5] = uncertainty_loss.item()

            self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(epoch)
            self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(epoch)

            if lr_changed:
                opt.param_groups[0]['lr'] = self.learning_rates[0]
                opt.param_groups[1]['lr'] = self.learning_rates[1]
                opt.param_groups[2]['lr'] = self.learning_rates[2]
                opt.param_groups[3]['lr'] = self.learning_rates[3]
                opt.param_groups[4]['lr'] = self.learning_rates[4]


            if epoch > 100 and (epoch % 10 == 0):
                if valid_losses.mean() < best_loss:
                    best_loss = valid_losses.mean()
                    best_net_state = deepcopy(net.state_dict())
                    best_opt_state = deepcopy(opt.state_dict())


            if (epoch+1)%20 == 0:
                logging.critical('[Seed{}] Epoch {} : forward loss {:.3e}, backward loss {:.3e}, reinference loss {:.3e}, pi_loss {:.3e}, valid_loss {:.3e} [TRAIN]'.format(self.seed, epoch, *losses[epoch, :4], valid_losses.mean()))

                plt.figure()
                for idx, label, color in zip(range(4), ['forward', 'backward', 'reinference', 'path integration'], ['r', 'b', 'g', 'm']):
                    plt.semilogy(losses[:epoch, idx], color=color, label=label, alpha=.8)
                plt.legend()
                plt.xlabel("Training step")
                plt.ylabel("Value of losses")
                plt.savefig(net.save_folder+'losses.pdf')
                plt.close('all')



            if (epoch+1) % 500 == 0 or (epoch == self.n_epochs) :
                tch.save(best_net_state, net.save_folder+'best_net.tch')
                np.savetxt(self.network.save_folder + 'losses.txt', losses)

                if self.do_resets:
                    net = network_register[self.net_name](**self.net_options)
                    net.load_state_dict(best_net_state)

                    opt = self.optimizer([
                                            {'params': net.gating_module.parameters()},
                                            {'params': net.representation_module.parameters()},
                                            {'params': net.z_encoder_module.parameters()},
                                            {'params': net.backward_module.parameters()},
                                            {'params': net.forward_module.parameters()},
                                        ])

                    opt.load_state_dict(best_opt_state)
                    opt.param_groups[0]['lr'] = self.learning_rates[0]
                    opt.param_groups[1]['lr'] = self.learning_rates[1]
                    opt.param_groups[2]['lr'] = self.learning_rates[2]
                    opt.param_groups[3]['lr'] = self.learning_rates[3]
                    opt.param_groups[4]['lr'] = self.learning_rates[4]

        tch.save(best_net_state, net.save_folder+'best_net.tch')
        tch.save(net.state_dict(), net.save_folder+'final_net.tch')
        np.savetxt(self.network.save_folder + 'losses.txt', losses)
        self.do_tests('final', net, env)


class EndToEndTrainerOffshelf:
    def __init__(self, params, seed):
        logging.critical(params)
        net_pars = params['net_params']
        env_pars = params['env_params']
        train_pars = params['train_params']
        policy_pars = params['policy_params']
        load_from = params['load_from']

        optimizer_params_scheduler_name = train_pars['optimizer_params_scheduler_name']
        optimizer_params_scheduler_options =  train_pars['optimizer_params_scheduler_options']
        # losses_weights_scheduler_name = train_pars['losses_weights_scheduler_name']
        # losses_weights_scheduler_options =  train_pars['losses_weights_scheduler_options']

        self.test_suite = params['test_suite']
        self.optimizer_name = train_pars['optimizer_name']

        if self.optimizer_name == 'Adam':
            self.optimizer = Adam
        elif self.optimizer_name == 'SGD':
            self.optimizer = SGD

        self.do_resets = train_pars['do_resets']
        self.n_epochs = train_pars['n_epochs']
        # self.batch_size = train_pars['batch_size']

        self.seed = seed
        self.policy = policy_register[policy_pars['type']](seed=seed, **policy_pars)

        self.net_name = net_pars['net_name']
        self.net_options = net_pars['options']
        self.network = network_register[self.net_name](**self.net_options) # NOTE: useful to define it here so that we can do the initialization and forget about load_rom, but not that relevant

        self.lr_scheduler = lr_scheduler_list[optimizer_params_scheduler_name](**optimizer_params_scheduler_options)
        # self.weights_scheduler = weights_scheduler_list[losses_weights_scheduler_name](**losses_weights_scheduler_options)

        # self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(-1)
        self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(-1)

        if load_from is not None:
            self.network.load_state_dict(tch.load(load_from+'seed{}/best_net.tch'.format(seed)), strict=False)

        logging.critical(env_pars)
        self.env = FixedRewardWorld(**env_pars, seed=seed)
        # self.tuple_loss = TransitionLoss(self.policy, **train_pars['tuple_loss_options'])
        self.pi_loss = PathIntegrationLoss(self.policy, **train_pars['pi_loss_options'])

        self.validation_loss = PathIntegrationLoss(self.policy,
                                                    batch_size=train_pars['pi_loss_options']['batch_size'],
                                                    epoch_len=train_pars['pi_loss_options']['epoch_len'],
                                                    im_availability=train_pars['pi_loss_options']['im_availability'],
                                                    corruption_rate=train_pars['pi_loss_options']['corruption_rate'],
                                                    additive_noise=0.)

        logging.info('Done initializing [seed{}]'.format(seed))

    def do_tests(self, epoch, net, env):
        if (epoch == 0) or (self.test_suite is None):
            return
        for test_name, test_params in self.test_suite.items():
            try:
                if epoch == 'final':
                    raise TypeError
                should_launch = (epoch % test_params['period'] == 0)
            except TypeError:
                should_launch = True

            if should_launch:
                logging.info('Launching test {}'.format(test_name))
                try:
                    os.makedirs(net.save_folder + '{}/'.format(epoch), exist_ok=True)
                except FileExistsError:
                    pass
                tests_register[test_name](net, env, test_params, epoch)
                logging.info('Done running test {}'.format(test_name))

    def train(self):
        logging.info('Starting training [seed{}]'.format(self.seed))
        net = self.network
        env = self.env
        logging.critical(net)

        opt = self.optimizer([
                                # {'params': net.gating_module.parameters(), 'lr': self.learning_rates[0]},
                                {'params': net.recurrence_module.parameters(), 'lr': self.learning_rates[0]},
                                {'params': net.representation_module.parameters(), 'lr': self.learning_rates[1]},
                                {'params': net.z_encoder_module.parameters(), 'lr': self.learning_rates[2]},
                                {'params': net.backward_module.parameters(), 'lr': self.learning_rates[3]},
                                # {'params': net.forward_module.parameters(), 'lr': self.learning_rates[4]},
                            ])

        losses = np.zeros((self.n_epochs+1,6))

        best_net = deepcopy(net)
        best_loss = 1e10

        valid_losses = np.zeros(10)
        for epoch in range(self.n_epochs+1):
            # Compute the losses
            # fwd_loss, bkw_loss, reinf_loss = self.tuple_loss(net, env)
            pi_loss, gating_loss, uncertainty_loss = self.pi_loss(net, env)

            if epoch % 10 == 0:
                with tch.set_grad_enabled(False):
                    valid_loss, _, _ = self.validation_loss(net, env)
                    valid_losses[(epoch%100)//10] = valid_loss.item()

            opt.zero_grad()
            tot_loss = pi_loss
            tot_loss.backward()
            opt.step()

            self.do_tests(epoch, net, env)
            losses[epoch, 0] = pi_loss.item()
            # losses[epoch, 1] = bkw_loss.item()
            # losses[epoch, 2] = reinf_loss.item()
            # losses[epoch, 3] = pi_loss.item()
            # losses[epoch, 4] = gating_loss.item()
            # losses[epoch, 5] = uncertainty_loss.item()

            # self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(epoch)
            self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(epoch)

            if lr_changed:
                opt.param_groups[0]['lr'] = self.learning_rates[0]
                opt.param_groups[1]['lr'] = self.learning_rates[1]
                opt.param_groups[2]['lr'] = self.learning_rates[2]
                opt.param_groups[3]['lr'] = self.learning_rates[3]
                # opt.param_groups[4]['lr'] = self.learning_rates[4]


            if epoch > 100 and (epoch % 10 == 0):
                if valid_losses.mean() < best_loss:
                    best_loss = valid_losses.mean()
                    best_net_state = deepcopy(net.state_dict())
                    best_opt_state = deepcopy(opt.state_dict())


            if (epoch+1)%20 == 0:
                logging.critical('[Seed{}] Epoch {} : PI loss {:.3e}, valid_loss {:.3e} [TRAIN]'.format(self.seed, epoch, losses[epoch, 0], valid_losses.mean()))

                plt.figure()
                for idx, label, color in zip(range(4), ['forward', 'backward', 'reinference', 'path integration'], ['r', 'b', 'g', 'm']):
                    plt.semilogy(losses[:epoch, idx], color=color, label=label, alpha=.8)
                plt.legend()
                plt.xlabel("Training step")
                plt.ylabel("Value of losses")
                plt.savefig(net.save_folder+'losses.pdf')
                plt.close('all')



            if (epoch+1) % 500 == 0 or (epoch == self.n_epochs) :
                tch.save(best_net_state, net.save_folder+'best_net.tch')
                np.savetxt(self.network.save_folder + 'losses.txt', losses)

                if self.do_resets:
                    net = network_register[self.net_name](**self.net_options)
                    net.load_state_dict(best_net_state)

                    opt = self.optimizer([
                                            # {'params': net.gating_module.parameters()},
                                            {'params': net.recurrence_module.parameters()},
                                            {'params': net.representation_module.parameters()},
                                            {'params': net.z_encoder_module.parameters()},
                                            {'params': net.backward_module.parameters()},
                                            # {'params': net.forward_module.parameters()},
                                        ])

                    opt.load_state_dict(best_opt_state)
                    opt.param_groups[0]['lr'] = self.learning_rates[0]
                    opt.param_groups[1]['lr'] = self.learning_rates[1]
                    opt.param_groups[2]['lr'] = self.learning_rates[2]
                    opt.param_groups[3]['lr'] = self.learning_rates[3]
                    # opt.param_groups[4]['lr'] = self.learning_rates[4]

        tch.save(best_net_state, net.save_folder+'best_net.tch')
        tch.save(net.state_dict(), net.save_folder+'final_net.tch')
        np.savetxt(self.network.save_folder + 'losses.txt', losses)
        self.do_tests('final', net, env)


class EndToEndTrainerHybrid:
    def __init__(self, params, seed):
        logging.critical(params)
        net_pars = params['net_params']
        env_pars = params['env_params']
        train_pars = params['train_params']
        policy_pars = params['policy_params']
        load_from = params['load_from']

        optimizer_params_scheduler_name = train_pars['optimizer_params_scheduler_name']
        optimizer_params_scheduler_options =  train_pars['optimizer_params_scheduler_options']
        losses_weights_scheduler_name = train_pars['losses_weights_scheduler_name']
        losses_weights_scheduler_options =  train_pars['losses_weights_scheduler_options']

        self.test_suite = params['test_suite']
        self.optimizer_name = train_pars['optimizer_name']

        if self.optimizer_name == 'Adam':
            self.optimizer = Adam
        elif self.optimizer_name == 'SGD':
            self.optimizer = SGD

        self.do_resets = train_pars['do_resets']
        self.n_epochs = train_pars['n_epochs']
        # self.batch_size = train_pars['batch_size']

        self.seed = seed
        self.policy = policy_register[policy_pars['type']](seed=seed, **policy_pars)

        self.net_name = net_pars['net_name']
        self.net_options = net_pars['options']
        self.network = network_register[self.net_name](**self.net_options) # NOTE: useful to define it here so that we can do the initialization and forget about load_rom, but not that relevant

        self.lr_scheduler = lr_scheduler_list[optimizer_params_scheduler_name](**optimizer_params_scheduler_options)
        self.weights_scheduler = weights_scheduler_list[losses_weights_scheduler_name](**losses_weights_scheduler_options)

        self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(-1)
        self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(-1)

        if load_from is not None:
            self.network.load_state_dict(tch.load(load_from+'seed{}/best_net.tch'.format(seed)), strict=False)

        logging.critical(env_pars)
        self.env = FixedRewardWorld(**env_pars, seed=seed)
        self.tuple_loss = TransitionLoss(self.policy, **train_pars['tuple_loss_options'])
        self.pi_loss = PathIntegrationLoss(self.policy, **train_pars['pi_loss_options'])

        self.validation_loss = PathIntegrationLoss(self.policy,
                                                    batch_size=train_pars['pi_loss_options']['batch_size'],
                                                    epoch_len=train_pars['pi_loss_options']['epoch_len'],
                                                    im_availability=train_pars['pi_loss_options']['im_availability'],
                                                    corruption_rate=train_pars['pi_loss_options']['corruption_rate'],
                                                    additive_noise=0.)

        logging.info('Done initializing [seed{}]'.format(seed))

    def do_tests(self, epoch, net, env):
        if (epoch == 0) or (self.test_suite is None):
            return
        for test_name, test_params in self.test_suite.items():
            try:
                if epoch == 'final':
                    raise TypeError
                should_launch = (epoch % test_params['period'] == 0)
            except TypeError:
                should_launch = True

            if should_launch:
                logging.info('Launching test {}'.format(test_name))
                try:
                    os.makedirs(net.save_folder + '{}/'.format(epoch), exist_ok=True)
                except FileExistsError:
                    pass
                tests_register[test_name](net, env, test_params, epoch)
                logging.info('Done running test {}'.format(test_name))

    def train(self):
        logging.info('Starting training [seed{}]'.format(self.seed))
        net = self.network
        env = self.env
        logging.critical(net)

        opt = self.optimizer([
                                {'params': net.recurrence_module.parameters(), 'lr': self.learning_rates[0]},
                                {'params': net.representation_module.parameters(), 'lr': self.learning_rates[1]},
                                {'params': net.z_encoder_module.parameters(), 'lr': self.learning_rates[2]},
                                {'params': net.backward_module.parameters(), 'lr': self.learning_rates[3]},
                                {'params': net.forward_module.parameters(), 'lr': self.learning_rates[4]},
                            ])

        losses = np.zeros((self.n_epochs+1,6))

        best_net = deepcopy(net)
        best_loss = 1e10

        valid_losses = np.zeros(10)
        for epoch in range(self.n_epochs+1):
            # Compute the losses
            fwd_loss, bkw_loss, reinf_loss = self.tuple_loss(net, env)
            pi_loss, gating_loss, uncertainty_loss = self.pi_loss(net, env)

            if epoch % 10 == 0:
                with tch.set_grad_enabled(False):
                    valid_loss, _, _ = self.validation_loss(net, env)
                    valid_losses[(epoch%100)//10] = valid_loss.item()

            opt.zero_grad()
            tot_loss = self.losses_weights[0] * bkw_loss + self.losses_weights[1] * fwd_loss + self.losses_weights[2] *reinf_loss
            tot_loss = tot_loss +  self.losses_weights[3] * pi_loss  + self.losses_weights[4] * gating_loss + self.losses_weights[5] * uncertainty_loss
            tot_loss.backward()
            opt.step()

            self.do_tests(epoch, net, env)
            losses[epoch, 0] = fwd_loss.item()
            losses[epoch, 1] = bkw_loss.item()
            losses[epoch, 2] = reinf_loss.item()
            losses[epoch, 3] = pi_loss.item()
            losses[epoch, 4] = gating_loss.item()
            losses[epoch, 5] = uncertainty_loss.item()


            if epoch % 10 == 0:
                with tch.set_grad_enabled(False):
                    valid_loss, _, _ = self.validation_loss(net, env)
                    valid_losses[(epoch%100)//10] = valid_loss.item()

            self.losses_weights, weights_changed = self.weights_scheduler.get_current_weights(epoch)
            self.learning_rates, lr_changed = self.lr_scheduler.get_current_lr(epoch)

            if lr_changed:
                opt.param_groups[0]['lr'] = self.learning_rates[0]
                opt.param_groups[1]['lr'] = self.learning_rates[1]
                opt.param_groups[2]['lr'] = self.learning_rates[2]
                opt.param_groups[3]['lr'] = self.learning_rates[3]
                opt.param_groups[4]['lr'] = self.learning_rates[4]


            if epoch > 100 and (epoch % 10 == 0):
                if valid_losses.mean() < best_loss:
                    best_loss = valid_losses.mean()
                    best_net_state = deepcopy(net.state_dict())
                    best_opt_state = deepcopy(opt.state_dict())


            if epoch%20 == 0:
                logging.critical('[Seed{}] Epoch {} : forward loss {:.3e}, backward loss {:.3e}, reinference loss {:.3e}, pi_loss {:.3e}, valid_loss {:.3e} [TRAIN]'.format(self.seed, epoch, *losses[epoch, :4], valid_losses.mean()))

                plt.figure()
                for idx, label, color in zip(range(4), ['forward', 'backward', 'reinference', 'path integration'], ['r', 'b', 'g', 'm']):
                    plt.semilogy(losses[:epoch, idx], color=color, label=label, alpha=.8)
                plt.legend()
                plt.xlabel("Training step")
                plt.ylabel("Value of losses")
                plt.savefig(net.save_folder+'losses.pdf')
                plt.close('all')


            if (epoch+1) % 500 == 0 or (epoch == self.n_epochs) :
                tch.save(best_net_state, net.save_folder+'best_net.tch')
                np.savetxt(self.network.save_folder + 'losses.txt', losses)

                if self.do_resets:
                    net = network_register[self.net_name](**self.net_options)
                    net.load_state_dict(best_net_state)

                    opt = self.optimizer([
                                            # {'params': net.gating_module.parameters()},
                                            {'params': net.recurrence_module.parameters()},
                                            {'params': net.representation_module.parameters()},
                                            {'params': net.z_encoder_module.parameters()},
                                            {'params': net.backward_module.parameters()},
                                            {'params': net.forward_module.parameters()},
                                        ])

                    opt.load_state_dict(best_opt_state)
                    opt.param_groups[0]['lr'] = self.learning_rates[0]
                    opt.param_groups[1]['lr'] = self.learning_rates[1]
                    opt.param_groups[2]['lr'] = self.learning_rates[2]
                    opt.param_groups[3]['lr'] = self.learning_rates[3]
                    opt.param_groups[4]['lr'] = self.learning_rates[4]

        tch.save(best_net_state, net.save_folder+'best_net.tch')
        tch.save(net.state_dict(), net.save_folder+'final_net.tch')
        np.savetxt(self.network.save_folder + 'losses.txt', losses)
        self.do_tests('final', net, env)
