import torch as tch
import torch
from torch.nn import Module, ModuleList
import torch.nn as nn
import os
import logging
# from reimplementations import *
import torch.nn.functional as F
from torch.nn import Parameter
from copy import deepcopy


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
        self.hidden_size = representation_size
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
            h, h_forward, h_observed, log_g = self.update_internal_state(h, image_representations[:,t+1], z_representations[:, t])
            # Decode the position
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

    def update_internal_state(self, state, im_rep, z_rep):
        # 1. Propose a new state using the forward model
        h_forward = self.forward_model(state, z_rep)
        # 2. Propose a new state from the image
        h_observed = im_rep
        # 3. Choose between the two
        log_g = self.gating_module(tch.cat([tch.zeros_like(h_observed), h_observed], dim=1))

        g = tch.exp(log_g)
        h = g[:, 0].unsqueeze(1) * h_forward + g[:, 1].unsqueeze(1) * h_observed
        return h, h_forward, h_observed, log_g

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
        self.hidden_size = representation_size
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

# Used to be in representations, but moved it back here
class GRU(torch.nn.Module):
    # NOTE: moving stuff to cuda will be taken care of by encapsulating module in networks.py (hopefully)
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=False):
        super(GRU, self).__init__()

        if num_layers > 1:
            raise RuntimeError('See-through reimplementations do not handle multi-layers')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        self.weight_ih_l0 = Parameter(torch.Tensor(3*self.hidden_size, self.input_size), requires_grad=True)
        self.weight_hh_l0 = Parameter(torch.Tensor(3*self.hidden_size, self.hidden_size), requires_grad=True)

        if not self.bias:
            self.bias_ih_l0 = Parameter(torch.zeros(3*self.hidden_size), requires_grad=True)
            self.bias_hh_l0 = Parameter(torch.zeros(3*self.hidden_size), requires_grad=True)
        else:
            self.bias_ih_l0 = Parameter(torch.Tensor(3*self.hidden_size), requires_grad=True)
            self.bias_hh_l0 = Parameter(torch.Tensor(3*self.hidden_size), requires_grad=True)


    def GRUCell(self, input, hidden, return_internals=False):
        gi = F.linear(input, self.weight_ih_l0, self.bias_ih_l0)
        gh = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)

        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        if not return_internals:
            return hy, None, None, None
        else:
            return hy, resetgate, inputgate, newgate

    def forward(self, input, hidden=None, batch_first=None, return_internals=False):
        if batch_first is None:
            batch_first = self.batch_first

        if batch_first:
            input = input.transpose(0, 1)

        epoch_len = input.shape[0]
        batch_size = input.shape[1]

        output = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)


        if return_internals:
            resetgates = torch.zeros(epoch_len, batch_size, self.hidden_size)
            inputgates = torch.zeros(epoch_len, batch_size, self.hidden_size)
            newgates = torch.zeros(epoch_len, batch_size, self.hidden_size)


        for step in range(epoch_len):
            hidden, resetgate, inputgate, newgate = self.GRUCell(input[step], hidden, return_internals=return_internals)
            output[step] = hidden
            if return_internals:
                resetgates[step] = resetgate
                inputgates[step] = inputgate
                newgates[step] = newgate

        if batch_first:
            output = output.transpose(0, 1)
            if return_internals:
                resetgates = resetgates.transpose(0, 1)
                inputgates = inputgates.transpose(0, 1)
                newgates = newgates.transpose(0, 1)

        if not return_internals:
            return output, hidden
        else:
            return output, hidden, resetgates, inputgates, newgates

class LSTM(torch.nn.Module):
    # NOTE: moving stuff to cuda will be taken care of by encapsulating module in networks.py (hopefully)
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, bias=True):
        super(LSTM, self).__init__()
        if num_layers > 1:
            raise RuntimeError('See-through reimplementations do not handle multi-layers')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        self.weight_ih_l0 = Parameter(torch.Tensor(4*self.hidden_size, self.input_size), requires_grad=True)
        self.weight_hh_l0 = Parameter(torch.Tensor(4*self.hidden_size, self.hidden_size), requires_grad=True)

        if not self.bias:
            self.bias_ih_l0 = None
            self.bias_hh_l0 = None
        else:
            self.bias_ih_l0 = Parameter(torch.Tensor(4*self.hidden_size), requires_grad=True)
            self.bias_hh_l0 = Parameter(torch.Tensor(4*self.hidden_size), requires_grad=True)


    def LSTMCell(self, input, hidden, return_internals=False):
        hx, cx = hidden
        gates = F.linear(input, self.weight_ih_l0, self.bias_ih_l0) + F.linear(hx, self.weight_hh_l0, self.bias_hh_l0)

        # ingate, forgetgate, cellgate, outgate = gates.squeeze(1).chunk(4, 1)
        ingate, forgetgate, cellgate, outgate = gates.squeeze(1).chunk(4, 1) # Maybe the order is fucked?

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        if not return_internals:
            return hy, cy, None, None, None, None
        else:
            return hy, cy, ingate, forgetgate, cellgate, outgate


    def forward(self, input, hidden=None, batch_first=None, return_internals=False):
        if batch_first is None:
            batch_first = self.batch_first

        # print('input shape, batch_first in forward: {}, {}'.format(input.shape, batch_first))
        if batch_first:
            input = input.transpose(0, 1)

        # print('input shape in forward afte batch_first transposition: {}'.format(input.shape))
        epoch_len = input.shape[0]
        batch_size = input.shape[1]

        if hidden is None:
            hidden = (torch.zeros(batch_size, self.hidden_size).to(self.device), torch.zeros(batch_size, self.hidden_size).to(self.device))

        output = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)
        if return_internals:
            ingates = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)
            forgetgates = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)
            cellgates = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)
            outgates = torch.zeros(epoch_len, batch_size, self.hidden_size).to(self.device)


        for step in range(epoch_len):
            # print('step{}'.format(step), input.shape, input[step].shape, hidden[0].shape, hidden[1].shape)
            hy, cy, ingate, forgetgate, cellgate, outgate = self.LSTMCell(input[step], hidden, return_internals=return_internals)
            output[step] = hy
            hidden = (hy, cy)
            if return_internals:
                ingates[step] = ingate
                forgetgates[step] = forgetgate
                cellgates[step] = cellgate
                outgates[step] = outgate

        if batch_first:
            output = output.transpose(0, 1)
            if return_internals:
                ingates = ingates.transpose(0, 1)
                forgetgates = forgetgates.transpose(0, 1)
                cellgates = cellgates.transpose(0, 1)
                outgates = outgates.transpose(0, 1)

        if not return_internals:
            return output, (hy, cy)
        else:
            return output, (hy, cy), ingates, forgetgates, cellgates, outgates

class LegacyReimplementationPathIntegrator(Module):
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_from=None, recurrence_type='GRU', recurrence_args={'hidden_size':None, 'num_layers': 1},
                    use_start_rep_explicitly=True, use_reimplementation=False, **kwargs):
        pass


class BigReimplementationPathIntegrator(Module):
    def __init__(self, representation_size=512, seed=0, device_name='cuda', save_folder='out/tests/', load_encoders_from=None, load_from=None, recurrence_type='GRU', recurrence_args={'hidden_size':None, 'num_layers': 1},
                    use_start_rep_explicitly=True, use_reimplementation=False, **kwargs):
        super(BigReimplementationPathIntegrator, self).__init__()
        self.type = 'offshelf'
        self.seed = seed
        tch.manual_seed(seed)
        tch.cuda.manual_seed(seed)
        self.is_cheater = False
        self.recurrence_type = recurrence_type

        self.representation_size = representation_size
        self.load_from = load_from
        self.device_name = device_name
        self.save_folder = save_folder + 'seed{}/'.format(seed)
        self.use_attention = False
        self.use_start_rep_explicitly = use_start_rep_explicitly
        self.use_reimplementation = use_reimplementation

        self.device = tch.device(self.device_name)

        try:
            os.makedirs(self.save_folder, exist_ok=True)
        except FileExistsError:
            pass

        self.representation_module = BigBasicConv(out_size=self.representation_size, activation='relu')
        self.z_encoder_module = BasicDense(in_size=2, out_size=self.representation_size, intermediate_sizes=[256], activation='relu')

        if recurrence_args['hidden_size'] == None:
            recurrence_args['hidden_size'] = representation_size

        self.hidden_size = recurrence_args['hidden_size']

        if self.use_reimplementation:
            if self.recurrence_type == 'GRU':
                self.recurrence_module = GRU(input_size=2*representation_size, batch_first=True, **recurrence_args)
            elif self.recurrence_type == 'LSTM':
                self.recurrence_module = LSTM(input_size=2*representation_size, batch_first=True, **recurrence_args)
            self.recurrence_module.device = self.device
        else:

            if self.recurrence_type == 'GRU':
                self.recurrence_module = nn.GRU(input_size=2*representation_size, batch_first=True, **recurrence_args)
            elif self.recurrence_type == 'LSTM':
                self.recurrence_module = nn.LSTM(input_size=2*representation_size, batch_first=True, **recurrence_args)

        if not self.use_start_rep_explicitly:
            self.backward_module = BasicDense(in_size=self.hidden_size, out_size=2, intermediate_sizes=[256], activation='relu')
        else:
            self.backward_module = BasicDense(in_size=2*self.hidden_size, out_size=2, intermediate_sizes=[256], activation='relu')

        self.to(self.device)
        if load_from is not None:
            self.load(load_from)
        elif load_encoders_from is not None:
            pass
            self.load_encoders(load_encoders_from)
            # if self.use_reimplementation:
            #     raise RuntimeError('Reimplemented networks are not initialized and should only ever be used with existing weights')

    def __validate_images(self, images_batch):
        try:
            images_batch = tch.from_numpy(images_batch)
        except:
            pass

        assert len(images_batch.shape) == 3, "ZFBModuleConv.__validate_images expects batch of retina images"
        images_batch = images_batch.reshape(images_batch.shape[0], 64, 64, 3).permute(0, 3, 1, 2)

        return images_batch.float().to(self.device)

    def __validate_z(self, z_batch):
        try:
            z_batch = tch.from_numpy(z_batch)
        except:
            pass

        assert len(z_batch.shape) == 2, "ZFBModuleConv.__validate_z expects batch of vectors"

        return z_batch.float().to(self.device)

    def get_representation(self, images_batch):
        # Encode the (vectorized) images
        images_batch = self.__validate_images(images_batch)
        assert len(images_batch.shape) == 4, "ZFBModuleConv expects batch of images"
        return self.representation_module(images_batch)


    def get_z_encoding(self, z_batch):
        z_batch = self.__validate_z(z_batch)
        return self.z_encoder_module(z_batch)

    def backward_model(self, reps1, reps2):
        inputs = tch.cat([reps1, reps2], dim=-1) # Was 1 instead of -1, but seemed t work so maybe its the same here
        return self.backward_module(inputs)


    def do_path_integration(self, image_representations, z_representations, return_all=False):
        assert len(image_representations.shape) == 3 # expect (bs, T, rep_size)
        bs = image_representations.shape[0]
        T = image_representations.shape[1] - 1
        # logging.critical(image_representations.shape)
        if return_all:
            if not self.use_reimplementation:
                raise UserWarning('Need to use reimplementation of GRU/LSTM if we want to have access to the gates')

        # We have one more image than transitions; use the first image to initialize the network
        first_rep, image_representations = tch.split(image_representations, [1, image_representations.shape[1]-1], dim=1)

        if not return_all:

            if self.recurrence_type == 'LSTM':
                _, (h0, c0) = self.recurrence_module(tch.cat([first_rep, tch.zeros_like(z_representations[:,0]).unsqueeze(1)], dim=-1))
                # h0 = h0.squeeze(1)
                internal_states, (hn, cn) = self.recurrence_module(tch.cat([image_representations, z_representations], dim=-1), (h0, c0))

            # elif self.recurrence_type in ['GRU', 'RNN', 'SimplifiedGRU']:
            elif self.recurrence_type == 'GRU':
                _, h0 = self.recurrence_module(tch.cat([first_rep, tch.zeros_like(z_representations[:,0]).unsqueeze(1)], dim=-1))
                internal_states, hn = self.recurrence_module(tch.cat([image_representations, z_representations], dim=-1), h0)

            # Use the internal states to compute the actual distance from start
            if not self.use_start_rep_explicitly:
                outputs = self.backward_module(internal_states)
            else:
                outputs = self.backward_module(tch.cat([internal_states, first_rep.expand(internal_states.shape)], dim=-1))

            return outputs, tch.zeros(*outputs.shape), internal_states

        elif return_all:
            if self.recurrence_type == 'LSTM':
                _, (h0, c0) = self.recurrence_module(tch.cat([first_rep, tch.zeros_like(z_representations[:,0]).unsqueeze(1)], dim=-1), return_internals=False)
                # h0 = h0.squeeze(1)
                internal_states, (hn, cn), ingates, forgetgates, cellgates, outgates = self.recurrence_module(tch.cat([image_representations, z_representations], dim=-1), (h0, c0), return_internals=True)

            # elif self.recurrence_type in ['GRU', 'SimplifiedGRU']:
            elif self.recurrence_type == 'GRU':
                _, h0 = self.recurrence_module(tch.cat([first_rep, tch.zeros_like(z_representations[:,0]).unsqueeze(1)], dim=-1), return_internals=False)
                internal_states, hn, resetgates, inputgates, newgates = self.recurrence_module(tch.cat([image_representations, z_representations], dim=-1), h0, return_internals=True)

            if not self.use_start_rep_explicitly:
                outputs = self.decoder(internal_states)
            else:
                outputs = self.decoder(tch.cat([internal_states, first_rep.expand(internal_states.shape)], dim=-1))

            if self.recurrence_type == 'LSTM':
                return outputs, internal_states, ingates, forgetgates, cellgates, outgates
            # elif self.recurrence_type in ['GRU', 'SimplifiedGRU']:
            elif self.recurrence_type == 'GRU':
                return outputs, internal_states, resetgates, inputgates, newgates

    def save(self, suffix=''):
        tch.save(self.state_dict(), self.save_folder + 'state_{}.pt'.format(suffix))

    def load(self, path):
        logging.critical('Trying to load state_dict with keys {}'.format(list(tch.load(path).keys())))
        dict = tch.load(path)
        self.load_state_dict(dict, strict=False)

    def load_encoders(self, path):
        logging.critical('Trying to load only encoders from path {}'.format(path))
        dict = tch.load(path)
        tmp = deepcopy(dict)
        for key in tmp.keys():
            if key.split('.')[0] not in ['z_encoder_module', 'representation_module']:
                dict.pop(key)
        logging.critical('After filtering, keys remaining are {}'.format(dict.keys()))
        self.load_state_dict(dict, strict=False)


network_register = {'FBModule': FBModule, 'ResetNetwork': ResetNetwork,
                    'BigFBModule': BigFBModule, 'BigResetNetwork': BigResetNetwork,
                    'LegacyReimplementationPathIntegrator': LegacyReimplementationPathIntegrator,
                    'BigReimplementationPathIntegrator': BigReimplementationPathIntegrator,
                    }
