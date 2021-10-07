# This environment is significantly harder to train...

from SensorFusion.src.trainer import *
from copy import deepcopy
import gc
import logging
import os
import json


BASE_FOLDER = 'out/'
MAP = 'DoubleDonut'
LAYOUT = 'Ambiguous'
PATH = BASE_FOLDER + MAP + '_' + LAYOUT + '/long_experiment/'


env_pars = {'map_name': MAP, 'objects_layout_name': LAYOUT, 'scale': .5, 'chosen_reward_pos': 'Default'}
policy_pars = {'type': 'RandomPolicy', 'step_size': .5}
net_pars = {'net_name': 'FBModule', 'options': {'representation_size': 512}}
train_pars = {'optimizer_name': 'Adam', 'losses_weights': None, 'lr': 5e-4, 'n_epochs': 2500, 'batch_size': 256, 'epoch_len': 1, 'loss_options': {'batch_size': 256,}, 'do_resets': True}
test_suite = {
            'sanity_check_position': {'period': 2**20, 'resolution': 20},
            'sanity_check_representation': {'period': 2**20, 'resolution': 20},
            'error_evolution': {'batch_size': 64, 'n_trajs': 1024, 'epoch_len': 50, 'step_size':.5, 'im_availability': .1, 'resetting_type': 'fixed'}
            }


default_params = {
    'env_params': env_pars,
    'policy_params': policy_pars,
    'net_params': net_pars,
    'train_params': train_pars,
    'test_suite': test_suite,
    'load_from': None
}


reinference_errors = [0.]
image_availabilities = [.2]

load_path = PATH + 'minimal_model/all_losses/'


# Main table
all_conditions_names = ['minimal_model/all_losses/', 'minimal_model/no_fb_losses/', 'offshelf_LSTM/vanilla/', 'hybrid_LSTM/pretrained_high_fb/',]
all_net_names = ['BigResetNetwork', 'BigResetNetwork, 'BigReimplementationPathIntegrator', 'BigHybridPathIntegrator'']
all_use_start_rep = [False, False, False, None]
all_load_from = [None, None, None load_path]


# Secondary table
# all_conditions_names = ['hybrid_LSTM/scratch_high_fb/', 'offshelf_LSTM/pretrained/', 'offshelf_LSTM/use_start_rep_no_pretrain/', 'offshelf_LSTM/pretrained_no_start_rep/']
# all_net_names = ['BigHybridPathIntegrator', 'BigReimplementationPathIntegrator', 'BigReimplementationPathIntegrator', 'BigReimplementationPathIntegrator']
# all_use_start_rep = [None, True, True, False]
# all_load_from = [None, load_path, None, load_path]


all_params = zip(all_conditions_names, all_net_names, all_use_start_rep, all_load_from)



class do_all:
    def __init__(self):
        pass

    def __call__(self, seed):
        logging.critical('Launching experiment with seed {}'.format(seed))

        for cond_name, net_name, use_start_rep, load_from in all_params:
            for image_availability in image_availabilities:
                for reinference_error in reinference_errors:
                    params = deepcopy(default_params)
                    params['net_params']['net_name'] = net_name
                    params['net_params']['options']['seed'] = seed

                    if net_name == 'BigReimplementationPathIntegrator':
                        params['net_params']['options']['recurrence_type'] = 'LSTM'
                        params['net_params']['options']['recurrence_args'] = {'hidden_size': None, 'num_layers': 1}
                        params['net_params']['options']['use_reimplementation'] = False

                    params['net_params']['options']['load_from'] = None
                    params['net_params']['options']['use_start_rep_explicitly'] = use_start_rep

                    if load_from is not None:
                        params['net_params']['options']['load_encoders_from'] = load_from + 'error_{}_avail_{}/seed{}/best_net.tch'.format(reinference_error, image_availability, seed)
                    else:
                        params['net_params']['options']['load_encoders_from'] = None

                    params['net_params']['options']['save_folder'] = PATH + cond_name + 'error_{}_avail_{}/'.format(reinference_error, image_availability)

                    params['train_params']['do_resets'] = True
                    params['policy_params']['type'] = 'RandomPolicy'
                    params['policy_params']['step_size'] = .5
                    params['train_params']['pi_loss_options'] = {}
                    params['train_params']['tuple_loss_options'] = {}

                    params['train_params']['pi_loss_options']['epoch_len'] = 40
                    params['train_params']['pi_loss_options']['corruption_rate'] = .5
                    params['train_params']['pi_loss_options']['batch_size'] = 32
                    params['train_params']['pi_loss_options']['im_availability'] = image_availability
                    params['train_params']['pi_loss_options']['additive_noise'] = [.1, 0., 0., 0., 0.] # Use the several_noises trick to help find resetting...

                    if cond_name in [
                        'minimal_model/all_losses/', 'hybrid_LSTM/pretrained/', 'hybrid_LSTM/scratch/',
                        'hybrid_LSTM/pretrained_high_fb/', 'hybrid_LSTM/scratch_high_fb/',
                        'hybrid_LSTM/pretrained_low_fb/', 'hybrid_LSTM/scratch_low_fb/', ]:
                        params['train_params']['tuple_loss_options']['batch_size'] = 512
                    else:
                        params['train_params']['tuple_loss_options']['batch_size'] = 4
                    params['train_params']['tuple_loss_options']['use_full_space'] = False # NOTE: this means the tupleLoss uses real transitions only; any performance improvement at long distances comes from PI loss only

                    # These are the defaults: all learning rates equal, no loading, nothing changing during training
                    params['train_params']['optimizer_params_scheduler_options'] = {}
                    params['train_params']['losses_weights_scheduler_options'] = {}


                    params['train_params']['n_epochs'] = 4000

                    if net_name == 'BigReimplementationPathIntegrator':
                        if cond_name == 'offshelf_LSTM/pretrained_frozen/':
                            params['train_params']['optimizer_params_scheduler_name'] = 'offshelf_dont_touch_encoding'
                        else:
                            params['train_params']['optimizer_params_scheduler_name'] = 'offshelf_default'
                    elif net_name == 'BigHybridPathIntegrator':
                            params['train_params']['optimizer_params_scheduler_name'] = 'hybrid_default'
                            params['train_params']['losses_weights_scheduler_name'] = 'hybrid_default'
                            if cond_name in ['hybrid_LSTM/pretrained_low_fb/', 'hybrid_LSTM/scratch_low_fb/']:
                                params['train_params']['losses_weights_scheduler_name'] = 'hybrid_low_models'
                            elif cond_name in ['hybrid_LSTM/pretrained_high_fb/', 'hybrid_LSTM/scratch_high_fb/']:
                                params['train_params']['losses_weights_scheduler_name'] = 'hybrid_high_models'
                    else:
                        if cond_name == 'minimal_model/all_losses/':
                            params['train_params']['losses_weights_scheduler_name'] = 'default'
                            params['train_params']['optimizer_params_scheduler_name'] = 'default'
                        else:
                            params['train_params']['losses_weights_scheduler_name'] = 'default_no_fb'
                            params['train_params']['optimizer_params_scheduler_name'] = 'default_no_fb'

                    test_every = 500
                    params['test_suite'] = {}

                    os.makedirs(params['net_params']['options']['save_folder'], exist_ok=True)
                    with open(params['net_params']['options']['save_folder'] + 'full_params.json', 'w+') as f:
                        json.dump(params, f, indent=4)

                    if net_name == 'BigReimplementationPathIntegrator':
                        trainer = EndToEndTrainerOffshelf(params, seed)
                    elif params['net_params']['net_name'] == 'BigHybridPathIntegrator':
                        logging.critical('Trained HybridEndToEnd [seed{}]'.format(seed))
                        trainer = EndToEndTrainerHybrid(params, seed)
                    else:
                        trainer = EndToEndTrainer(params, seed)

                    trainer.train()
                    gc.collect()


if __name__ == '__main__':
    import matplotlib
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

    from multiprocessing_logging import install_mp_handler
    import multiprocessing

    n_threads = 4
    n_seeds = 4
    start_seed = 0

    multiprocessing.set_start_method('spawn')
    install_mp_handler()
    pool = multiprocessing.Pool(n_threads, initializer=install_mp_handler)
    pool.map(do_all(), range(start_seed, start_seed+n_seeds))
