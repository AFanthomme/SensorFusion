import torch as tch
from torch.nn import Module, ModuleList
import torch.nn as nn
import os
import logging

# TODO: add to resetnetwork a "step" function that takes current internal state and inputs and updates it (RNNCell style)

class BasicDense(Module):
    def __init__(self, in_size=1024, out_size=512, intermediate_sizes=[512, 512], activation='relu', use_bias=True):
        Module.__init__(self)
        self.sizes = [in_size] + intermediate_sizes + [out_size]
        self.n_layers = len(self.sizes)-1

        if activation == 'relu':
            self.non_linearity = nn.functional.relu
        else:
            raise RuntimeError('Invalid activation name')

        self.layers = ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1], bias=use_bias) for i in range(self.n_layers)])

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.non_linearity(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class DenseClassifier(Module):
    def __init__(self, in_size=1024, out_size=512, intermediate_sizes=[512, 512], activation='relu', use_bias=True):
        Module.__init__(self)
        self.sizes = [in_size] + intermediate_sizes + [out_size]
        self.n_layers = len(self.sizes)-1

        if activation == 'relu':
            self.non_linearity = nn.functional.relu
        else:
            raise RuntimeError('Invalid activation name')

        self.layers = ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1], bias=use_bias) for i in range(self.n_layers)])

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.non_linearity(self.layers[i](x))
        x = self.layers[-1](x)
        return nn.functional.log_softmax(x, dim=-1)



class BasicConv(Module):
    def __init__(self, out_size=512, activation='relu'):
        Module.__init__(self)
        self.out_size = out_size
        self.conv1 =  nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=2)
        self.conv2 =  nn.Conv2d(16, 32, kernel_size=5, stride=5, padding=2)
        self.fc1 = nn.Linear(800, 512)
        self.fc2 = nn.Linear(512, self.out_size)

        if activation == 'relu':
            self.non_linearity = nn.functional.relu
        else:
            raise RuntimeError('Invalid activation name')

    def forward(self, images_batch):
        assert len(images_batch.shape) == 4
        assert images_batch.shape[1] == 3
        out = self.non_linearity(self.conv1(images_batch))
        out = self.non_linearity(self.conv2(out))
        out = out.reshape(out.shape[0], -1)
        out = self.non_linearity(self.fc1(out))
        return self.fc2(out)


class BigBasicConv(Module):
    def __init__(self, out_size=512, activation='relu'):
        Module.__init__(self)
        self.out_size = out_size
        self.conv1 =  nn.Conv2d(3, 64, kernel_size=5, stride=3, padding=2)
        self.conv2 =  nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=2)
        self.conv3 =  nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=2)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.out_size)

        if activation == 'relu':
            self.non_linearity = nn.functional.relu
        else:
            raise RuntimeError('Invalid activation name')

    def forward(self, images_batch):
        assert len(images_batch.shape) == 4
        assert images_batch.shape[1] == 3
        out = self.non_linearity(self.conv1(images_batch))
        out = self.non_linearity(self.conv2(out))
        out = self.non_linearity(self.conv3(out))
        out = out.reshape(out.shape[0], -1)
        out = self.non_linearity(self.fc1(out))
        out = self.non_linearity(self.fc2(out))
        return self.fc3(out)



class FBModule(Module):
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_from=None, do_complete_init=True, **kwargs):
        Module.__init__(self)

        if not do_complete_init:
            return

        self.seed = seed
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)

        self.representation_size = representation_size
        self.load_from = load_from
        self.device_name = device_name
        self.save_folder = save_folder + 'seed{}/'.format(seed)
        self.device = tch.device(self.device_name)

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except FileExistsError:
            pass

        self.representation_module = BasicConv(out_size=self.representation_size, activation='relu')
        self.z_encoder_module = BasicDense(in_size=2, out_size=self.representation_size, intermediate_sizes=[256], activation='relu')
        self.backward_module = BasicDense(in_size=2*self.representation_size, intermediate_sizes=[256], out_size=2, activation='relu')
        self.forward_module = BasicDense(in_size=2*self.representation_size, out_size=self.representation_size, intermediate_sizes=[self.representation_size], activation='relu')

        self.to(self.device)
        if load_from is not None:
            self.load(load_from)

    def __validate_images(self, images_batch):
        try:
            images_batch = tch.from_numpy(images_batch)
        except:
            pass

        assert len(images_batch.shape) == 3, "FBModule.__validate_images expects batch of retina images"
        images_batch = images_batch.reshape(images_batch.shape[0], 64, 64, 3).permute(0, 3, 1, 2)

        return images_batch.float().to(self.device)

    def __validate_z(self, z_batch):
        try:
            z_batch = tch.from_numpy(z_batch)
        except:
            pass

        assert len(z_batch.shape) == 2, "FBModule.__validate_z expects batch of vectors"

        return z_batch.float().to(self.device)

    def get_representation(self, images_batch):
        # Encode the (vectorized) images
        images_batch = self.__validate_images(images_batch)
        assert len(images_batch.shape) == 4, "FBModule expects batch of images"
        return self.representation_module(images_batch)

    def get_z_encoding(self, z_batch):
        z_batch = self.__validate_z(z_batch)
        return self.z_encoder_module(z_batch)

    def backward_model(self, reps1, reps2):
        inputs = tch.cat([reps1, reps2], dim=1)
        return self.backward_module(inputs)

    def forward_model(self, reps1, actions_encoding):
        inputs = tch.cat([reps1, actions_encoding], dim=1)
        return self.forward_module(inputs)

    def save(self, suffix=''):
        tch.save(self.state_dict(), self.save_folder + 'state_{}.pt'.format(suffix))

    def load(self, path):
        self.load_state_dict(tch.load(path), strict=False)


class ResetNetwork(Module):
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_from=None, do_complete_init=True, **kwargs):
        Module.__init__(self)

        if not do_complete_init:
            return

        self.seed = seed
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)

        self.representation_size = representation_size
        self.load_from = load_from
        self.device_name = device_name
        self.save_folder = save_folder + 'seed{}/'.format(seed)
        self.device = tch.device(self.device_name)

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except FileExistsError:
            pass

        self.representation_module = BasicConv(out_size=self.representation_size, activation='relu')
        self.z_encoder_module = BasicDense(in_size=2, out_size=self.representation_size, intermediate_sizes=[256], activation='relu')
        self.backward_module = BasicDense(in_size=2*self.representation_size, intermediate_sizes=[256], out_size=2, activation='relu')
        self.gating_module = DenseClassifier(in_size=2*self.representation_size, out_size=2, intermediate_sizes=[self.representation_size // 4, 64], activation='relu')
        self.forward_module = BasicDense(in_size=2*self.representation_size, out_size=self.representation_size, intermediate_sizes=[self.representation_size], activation='relu')

        self.to(self.device)
        if load_from is not None:
            self.load(load_from)

    def __validate_images(self, images_batch):
        try:
            images_batch = tch.from_numpy(images_batch)
        except:
            pass

        assert len(images_batch.shape) == 3, "FBModule.__validate_images expects batch of retina images"
        images_batch = images_batch.reshape(images_batch.shape[0], 64, 64, 3).permute(0, 3, 1, 2)

        return images_batch.float().to(self.device)

    def __validate_z(self, z_batch):
        try:
            z_batch = tch.from_numpy(z_batch)
        except:
            pass

        assert len(z_batch.shape) == 2, "FBModule.__validate_z expects batch of vectors"

        return z_batch.float().to(self.device)

    def get_representation(self, images_batch):
        # Encode the (vectorized) images
        images_batch = self.__validate_images(images_batch)
        assert len(images_batch.shape) == 4, "FBModule expects batch of images"
        return self.representation_module(images_batch)


    def get_z_encoding(self, z_batch):
        z_batch = self.__validate_z(z_batch)
        return self.z_encoder_module(z_batch)

    def backward_model(self, reps1, reps2):
        inputs = tch.cat([reps1, reps2], dim=1)
        return self.backward_module(inputs)

    def forward_model(self, reps1, actions_encoding):
        inputs = tch.cat([reps1, actions_encoding], dim=1)
        return self.forward_module(inputs)


    def do_path_integration(self, image_representations, z_representations, return_all=False):
        assert len(image_representations.shape) == 3 # expect (bs, T, rep_size)
        bs = image_representations.shape[0]
        T = image_representations.shape[1] - 1
        internal_states = tch.zeros(bs, T, self.representation_size).to(self.device)
        outputs = tch.zeros(bs, T, 2).to(self.device)
        log_gatings =  tch.zeros(bs, T, 2).to(self.device)
        resets =  tch.zeros(bs, T).to(self.device)

        if return_all:
            outputs_forward = tch.zeros(bs, T, 2).to(self.device)
            outputs_visual = tch.zeros(bs, T, 2).to(self.device)

        # At the beginning, we force the network in the state given by its visual system
        h = image_representations[:, 0]

        # After that, update according to the following scheme:
        for t in range(T):
            # 1. Propose a new state using the forward model
            h_forward = self.forward_model(h, z_representations[:, t])
            # 2. Propose a new state from the image
            h_observed = image_representations[:,t+1]
            # 3. Choose between the two
            log_g = self.gating_module(tch.cat([tch.zeros_like(h_observed), h_observed], dim=1))

            g = tch.exp(log_g)
            h = g[:, 0].unsqueeze(1) * h_forward + g[:, 1].unsqueeze(1) * h_observed

            # 4. Decode the position
            out = self.backward_model(image_representations[:, 0], h)


            if return_all:
                outputs_forward[:, t] = self.backward_model(image_representations[:, 0], h_forward)
                outputs_visual[:, t] = self.backward_model(image_representations[:, 0], h_observed)

            internal_states[:, t] = h.clone()
            log_gatings[:, t] = log_g
            outputs[:, t] = out

        if return_all:
            return outputs, log_gatings, internal_states, outputs_forward, outputs_visual
        else:
            return outputs, log_gatings, internal_states


    def save(self, suffix=''):
        tch.save(self.state_dict(), self.save_folder + 'state_{}.pt'.format(suffix))

    def load(self, path):
        logging.critical('loading weights from : {}'.format(path))
        self.load_state_dict(tch.load(path), strict=False) # Need strict kwarg otherwise cannot load partial state_dict



class BigFBModule(FBModule):
    # Only difference with Dense is the representation module.
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_from=None, **kwargs):
        FBModule.__init__(self, do_complete_init=False)
        self.seed = seed
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)

        self.representation_size = representation_size
        self.load_from = load_from
        self.device_name = device_name
        self.save_folder = save_folder + 'seed{}/'.format(seed)
        self.device = tch.device(self.device_name)

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except FileExistsError:
            pass

        self.representation_module = BigBasicConv(out_size=self.representation_size, activation='relu')
        self.z_encoder_module = BasicDense(in_size=2, out_size=self.representation_size, intermediate_sizes=[256], activation='relu')
        self.backward_module = BasicDense(in_size=2*self.representation_size, intermediate_sizes=[256], out_size=2, activation='relu')
        self.forward_module = BasicDense(in_size=2*self.representation_size, out_size=self.representation_size, intermediate_sizes=[self.representation_size], activation='relu')

        self.to(self.device)
        if load_from is not None:
            self.load(load_from)



class BigResetNetwork(ResetNetwork):
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_from=None, **kwargs):
        ResetNetwork.__init__(self, do_complete_init=False)

        self.seed = seed
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)

        self.representation_size = representation_size
        self.load_from = load_from
        self.device_name = device_name
        self.save_folder = save_folder + 'seed{}/'.format(seed)
        self.device = tch.device(self.device_name)

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except FileExistsError:
            pass

        self.representation_module = BigBasicConv(out_size=self.representation_size, activation='relu')
        self.z_encoder_module = BasicDense(in_size=2, out_size=self.representation_size, intermediate_sizes=[256], activation='relu')
        self.backward_module = BasicDense(in_size=2*self.representation_size, intermediate_sizes=[256], out_size=2, activation='relu')
        self.gating_module = DenseClassifier(in_size=2*self.representation_size, out_size=2, intermediate_sizes=[self.representation_size // 4, 64], activation='relu')
        self.forward_module = BasicDense(in_size=2*self.representation_size, out_size=self.representation_size, intermediate_sizes=[self.representation_size], activation='relu')

        self.to(self.device)
        if load_from is not None:
            self.load(load_from)





network_register = {'FBModule': FBModule, 'ResetNetwork': ResetNetwork,
                    'BigFBModule':BigFBModule, 'BigResetNetwork': BigResetNetwork,
                    }
