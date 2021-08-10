from trainer import *
from copy import deepcopy
import gc
import logging
import os
import json


BASE_FOLDER = 'out/'
MAP = 'SnakePath'
LAYOUT = 'Default'
PATH = BASE_FOLDER + MAP + '_' + LAYOUT + '/'


env_pars = {'map_name': MAP, 'objects_layout_name': LAYOUT, 'scale': .5}
policy_pars = {'type': 'RandomPolicy', 'step_size': .5}
net_pars = {'net_name': 'FBModule', 'options': {'representation_size': 512}}
train_pars = {'optimizer_name': 'Adam', 'losses_weights': [1., 1., 1.], 'lr': 5e-4, 'n_epochs': 2500, 'batch_size': 256, 'epoch_len': 1, 'loss_options': {'batch_size': 256,}, 'do_resets': True}
test_suite = {
            'sanity_check_position': {'period': 2**20, 'resolution': 20},
            'sanity_check_representation': {'period': 2**20, 'resolution': 20},
            }


default_params = {
    'env_params': env_pars,
    'policy_params': policy_pars,
    'net_params': net_pars,
    'train_params': train_pars,
    'test_suite': test_suite,
    'load_from': None
}

class train_end_to_end:
    def __init__(self):
        pass

    def __call__(self, seed):
        logging.critical('Launching experiment with seed {}'.format(seed))

        protocols = ['default']
        # protocols = ['annealed_several_noises_001', 'annealed_several_noises_0025', 'default']

        for protocol in protocols:
            params = deepcopy(default_params)
            params['net_params']['net_name'] = 'BigResetNetwork' # Could use the non-big one, but it avoids boilerplate in environment which is already a nightmare
            params['net_params']['options']['seed'] = seed
            params['net_params']['options']['save_folder'] = PATH + 'end_to_end/{}/'.format(protocol)
            params['train_params']['n_epochs'] = 8000

            # params['train_params']['do_resets'] = False
            params['train_params']['do_resets'] = True

            params['policy_params']['type'] = 'RandomPolicy'
            params['policy_params']['step_size'] = .5

            params['train_params']['pi_loss_options'] = {}
            params['train_params']['tuple_loss_options'] = {}

            params['train_params']['pi_loss_options']['im_availability'] = .2
            # params['train_params']['pi_loss_options']['epoch_len'] = 30
            params['train_params']['pi_loss_options']['epoch_len'] = 15
            params['train_params']['pi_loss_options']['corruption_rate'] = .5
            params['train_params']['pi_loss_options']['batch_size'] = 32
            params['train_params']['tuple_loss_options']['batch_size'] = 128
            # params['train_params']['tuple_loss_options']['batch_size'] = 128
            params['train_params']['tuple_loss_options']['use_full_space'] = False # NOTE: this means the tupleLoss uses real transitions only; any performance improvement at long distances comes from PI loss only


            # These are the defaults: all learning rates equal, no loading, nothing changing during training
            params['net_params']['options']['load_from'] = None
            params['train_params']['optimizer_params_scheduler_options'] = {}
            params['train_params']['losses_weights_scheduler_options'] = {}


            # Set the params independantly, longer but easier to read
            logging.critical('In end-to-end trainer, using protocol {} [seed{}]'.format(protocol, seed))
            if protocol == 'default':
                params['train_params']['n_epochs'] = 4000
                params['train_params']['losses_weights_scheduler_name'] = 'default'
                params['train_params']['optimizer_params_scheduler_name'] = 'default'
            elif protocol == 'annealed_several_noises_001':
                params['train_params']['losses_weights_scheduler_name'] = 'annealed'
                params['train_params']['optimizer_params_scheduler_name'] = 'annealed'
                params['train_params']['pi_loss_options']['additive_noise'] = [0.01, 0., 0., 0., 0.]
            elif protocol == 'annealed_several_noises_0025':
                params['train_params']['losses_weights_scheduler_name'] = 'annealed'
                params['train_params']['optimizer_params_scheduler_name'] = 'annealed'
                params['train_params']['pi_loss_options']['additive_noise'] = [0.025, 0., 0., 0., 0.]

            logging.critical('In end-to-end trainer, use protocol {} and attempt to load weights from {} [seed{}]'.format(protocol, params['net_params']['options']['load_from'], seed))




            params['test_suite'] = {
                                    'sanity_check_position': {'period': 500, 'resolution': 20},
                                    'sanity_check_representation': {'period': 500, 'resolution': 20},
                                    'path_integrator_test': {'period': 500,
                                                                'epoch_len': params['train_params']['pi_loss_options']['epoch_len'],
                                                                'step_size': params['policy_params']['step_size'],
                                                                'im_availability': params['train_params']['pi_loss_options']['im_availability'],
                                                                'corruption_rate': params['train_params']['pi_loss_options']['corruption_rate'],
                                                                'policy_pars': params['policy_params']},
                                    }



            os.makedirs(params['net_params']['options']['save_folder'], exist_ok=True)
            with open(params['net_params']['options']['save_folder'] + 'full_params.json', 'w+') as f:
                json.dump(params, f, indent=4)

            trainer = EndToEndTrainer(params, seed)
            trainer.train()
            gc.collect()

if __name__ == '__main__':
    import matplotlib
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

    from multiprocessing_logging import install_mp_handler
    import multiprocessing

    n_threads = 2
    n_seeds = 2
    start_seed = 0

    multiprocessing.set_start_method('spawn')
    install_mp_handler()
    pool = multiprocessing.Pool(n_threads, initializer=install_mp_handler)
    pool.map(train_end_to_end(), range(start_seed, start_seed+n_seeds))
