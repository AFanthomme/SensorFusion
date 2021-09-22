CRITICAL:root:Launching experiment with seed 0
CRITICAL:root:Launching experiment with seed 1
CRITICAL:root:Trained HybridEndToEnd [seed0]
CRITICAL:root:{'env_params': {'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'scale': 0.5, 'chosen_reward_pos': 'Default'}, 'policy_params': {'type': 'RandomPolicy', 'step_size': 0.5}, 'net_params': {'net_name': 'BigHybridPathIntegrator', 'options': {'representation_size': 512, 'seed': 0, 'recurrence_type': 'LSTM', 'recurrence_args': {'hidden_size': None, 'num_layers': 1}, 'use_reimplementation': False, 'use_start_rep_explicitly': None, 'load_from': None, 'load_encoders_from': None, 'save_folder': 'out/DoubleDonut_Default/long_experiment/hybrid_LSTM/scratch/error_0.0_avail_0.2/'}}, 'train_params': {'optimizer_name': 'Adam', 'losses_weights': None, 'lr': 0.0005, 'n_epochs': 4000, 'batch_size': 256, 'epoch_len': 1, 'loss_options': {'batch_size': 256}, 'do_resets': True, 'pi_loss_options': {'epoch_len': 40, 'corruption_rate': 0.5, 'batch_size': 32, 'im_availability': 0.2, 'additive_noise': 0.0}, 'tuple_loss_options': {'batch_size': 512, 'use_full_space': False}, 'optimizer_params_scheduler_options': {}, 'losses_weights_scheduler_options': {}, 'optimizer_params_scheduler_name': 'hybrid_default', 'losses_weights_scheduler_name': 'hybrid_default'}, 'test_suite': {}, 'load_from': None}
CRITICAL:root:Trained HybridEndToEnd [seed1]
CRITICAL:root:{'env_params': {'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'scale': 0.5, 'chosen_reward_pos': 'Default'}, 'policy_params': {'type': 'RandomPolicy', 'step_size': 0.5}, 'net_params': {'net_name': 'BigHybridPathIntegrator', 'options': {'representation_size': 512, 'seed': 1, 'recurrence_type': 'LSTM', 'recurrence_args': {'hidden_size': None, 'num_layers': 1}, 'use_reimplementation': False, 'use_start_rep_explicitly': None, 'load_from': None, 'load_encoders_from': None, 'save_folder': 'out/DoubleDonut_Default/long_experiment/hybrid_LSTM/scratch/error_0.0_avail_0.2/'}}, 'train_params': {'optimizer_name': 'Adam', 'losses_weights': None, 'lr': 0.0005, 'n_epochs': 4000, 'batch_size': 256, 'epoch_len': 1, 'loss_options': {'batch_size': 256}, 'do_resets': True, 'pi_loss_options': {'epoch_len': 40, 'corruption_rate': 0.5, 'batch_size': 32, 'im_availability': 0.2, 'additive_noise': 0.0}, 'tuple_loss_options': {'batch_size': 512, 'use_full_space': False}, 'optimizer_params_scheduler_options': {}, 'losses_weights_scheduler_options': {}, 'optimizer_params_scheduler_name': 'hybrid_default', 'losses_weights_scheduler_name': 'hybrid_default'}, 'test_suite': {}, 'load_from': None}
CRITICAL:root:BasicDense(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=512, bias=True)
  )
)
CRITICAL:root:{'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'scale': 0.5, 'chosen_reward_pos': 'Default'}
CRITICAL:root:n_rooms in our env : 16
CRITICAL:root:Allowed starting rooms : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
CRITICAL:root:called world.set_seed
CRITICAL:root:{'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'seed': 0}
/home/fanthomme/miniconda3/lib/python3.8/site-packages/gym/logger.py:30: UserWarning: [33mWARN: Box bound precision lowered by casting to float32[0m
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
CRITICAL:root:Using path integration loss with, image_availability 0.2, corruption rate 0.5, additive_noise 0.0 [seed0]
CRITICAL:root:Using path integration loss with, image_availability 0.2, corruption rate 0.5, additive_noise 0.0 [seed0]
CRITICAL:root:BigHybridPathIntegrator(
  (representation_module): BigBasicConv(
    (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(3, 3), padding=(2, 2))
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(3, 3), padding=(2, 2))
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(3, 3), padding=(2, 2))
    (fc1): Linear(in_features=1024, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=512, bias=True)
  )
  (z_encoder_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=2, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=512, bias=True)
    )
  )
  (forward_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (backward_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=1024, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=2, bias=True)
    )
  )
  (recurrence_module): LSTM(1024, 512, batch_first=True)
)
CRITICAL:root:BasicDense(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=512, bias=True)
  )
)
CRITICAL:root:{'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'scale': 0.5, 'chosen_reward_pos': 'Default'}
CRITICAL:root:n_rooms in our env : 16
CRITICAL:root:Allowed starting rooms : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
CRITICAL:root:called world.set_seed
CRITICAL:root:{'map_name': 'DoubleDonut', 'objects_layout_name': 'Default', 'seed': 1}
/home/fanthomme/miniconda3/lib/python3.8/site-packages/gym/logger.py:30: UserWarning: [33mWARN: Box bound precision lowered by casting to float32[0m
  warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))
CRITICAL:root:Using path integration loss with, image_availability 0.2, corruption rate 0.5, additive_noise 0.0 [seed1]
CRITICAL:root:Using path integration loss with, image_availability 0.2, corruption rate 0.5, additive_noise 0.0 [seed1]
CRITICAL:root:BigHybridPathIntegrator(
  (representation_module): BigBasicConv(
    (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(3, 3), padding=(2, 2))
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(3, 3), padding=(2, 2))
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(3, 3), padding=(2, 2))
    (fc1): Linear(in_features=1024, out_features=1024, bias=True)
    (fc2): Linear(in_features=1024, out_features=512, bias=True)
    (fc3): Linear(in_features=512, out_features=512, bias=True)
  )
  (z_encoder_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=2, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=512, bias=True)
    )
  )
  (forward_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): Linear(in_features=512, out_features=512, bias=True)
    )
  )
  (backward_module): BasicDense(
    (layers): ModuleList(
      (0): Linear(in_features=1024, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=2, bias=True)
    )
  )
  (recurrence_module): LSTM(1024, 512, batch_first=True)
)
/home/fanthomme/riley/SensorFusion/losses.py:88: UserWarning: where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/aten/src/ATen/native/TensorCompare.cpp:255.)
  corrupt = tch.where(tch.bernoulli(self.corruption_rate * tch.ones(self.batch_size, self.epoch_len+1)).byte(), ims_to_perturb, tch.zeros(self.batch_size, self.epoch_len+1)).bool()
CRITICAL:root:[Seed1] Epoch 0 : forward loss 2.436e-03, backward loss 1.782e-01, reinference loss 1.783e-01, pi_loss 1.681e+00, valid_loss 1.529e-01 [TRAIN]
/home/fanthomme/riley/SensorFusion/losses.py:88: UserWarning: where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/aten/src/ATen/native/TensorCompare.cpp:255.)
  corrupt = tch.where(tch.bernoulli(self.corruption_rate * tch.ones(self.batch_size, self.epoch_len+1)).byte(), ims_to_perturb, tch.zeros(self.batch_size, self.epoch_len+1)).bool()
CRITICAL:root:[Seed0] Epoch 0 : forward loss 2.529e-03, backward loss 1.680e-01, reinference loss 1.674e-01, pi_loss 1.590e+00, valid_loss 1.332e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 20 : forward loss 2.706e-03, backward loss 1.668e-01, reinference loss 1.825e-01, pi_loss 1.277e+00, valid_loss 4.070e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 20 : forward loss 3.044e-03, backward loss 1.519e-01, reinference loss 1.525e-01, pi_loss 1.422e+00, valid_loss 4.249e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 40 : forward loss 3.787e-03, backward loss 1.597e-01, reinference loss 1.471e-01, pi_loss 3.607e-01, valid_loss 5.407e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 40 : forward loss 2.323e-03, backward loss 1.402e-01, reinference loss 1.590e-01, pi_loss 1.032e+00, valid_loss 5.481e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 60 : forward loss 2.110e-03, backward loss 1.322e-01, reinference loss 1.092e-01, pi_loss 2.225e-01, valid_loss 5.980e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 60 : forward loss 2.858e-03, backward loss 1.237e-01, reinference loss 1.178e-01, pi_loss 3.263e-01, valid_loss 6.389e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 80 : forward loss 1.004e-03, backward loss 1.109e-01, reinference loss 9.725e-02, pi_loss 1.471e-01, valid_loss 6.257e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 80 : forward loss 1.297e-03, backward loss 1.028e-01, reinference loss 8.924e-02, pi_loss 1.581e-01, valid_loss 6.767e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 100 : forward loss 5.558e-04, backward loss 1.070e-01, reinference loss 9.134e-02, pi_loss 4.508e-02, valid_loss 4.860e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 100 : forward loss 6.921e-04, backward loss 9.826e-02, reinference loss 7.911e-02, pi_loss 5.142e-02, valid_loss 5.647e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 120 : forward loss 5.147e-04, backward loss 9.533e-02, reinference loss 7.166e-02, pi_loss 4.503e-02, valid_loss 3.008e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 120 : forward loss 4.667e-04, backward loss 9.112e-02, reinference loss 6.407e-02, pi_loss 5.823e-02, valid_loss 2.266e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 140 : forward loss 4.257e-04, backward loss 8.804e-02, reinference loss 6.816e-02, pi_loss 3.536e-02, valid_loss 1.687e-01 [TRAIN]
CRITICAL:root:[Seed1] Epoch 140 : forward loss 3.722e-04, backward loss 7.296e-02, reinference loss 5.794e-02, pi_loss 2.507e-02, valid_loss 1.174e-01 [TRAIN]
CRITICAL:root:[Seed0] Epoch 160 : forward loss 4.106e-04, backward loss 7.626e-02, reinference loss 4.819e-02, pi_loss 2.055e-02, valid_loss 8.351e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 160 : forward loss 4.431e-04, backward loss 7.216e-02, reinference loss 4.187e-02, pi_loss 6.219e-02, valid_loss 7.300e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 180 : forward loss 3.693e-04, backward loss 7.087e-02, reinference loss 3.717e-02, pi_loss 3.042e-02, valid_loss 5.093e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 180 : forward loss 3.396e-04, backward loss 6.168e-02, reinference loss 3.211e-02, pi_loss 2.061e-02, valid_loss 6.209e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 200 : forward loss 3.761e-04, backward loss 6.180e-02, reinference loss 3.198e-02, pi_loss 1.583e-02, valid_loss 3.452e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 200 : forward loss 3.753e-04, backward loss 5.633e-02, reinference loss 3.717e-02, pi_loss 8.007e-02, valid_loss 5.839e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 220 : forward loss 3.890e-04, backward loss 5.129e-02, reinference loss 2.082e-02, pi_loss 3.742e-02, valid_loss 2.909e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 220 : forward loss 3.167e-04, backward loss 4.494e-02, reinference loss 1.624e-02, pi_loss 4.312e-02, valid_loss 5.341e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 240 : forward loss 2.929e-04, backward loss 4.500e-02, reinference loss 1.274e-02, pi_loss 2.047e-02, valid_loss 2.684e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 240 : forward loss 3.209e-04, backward loss 4.147e-02, reinference loss 1.613e-02, pi_loss 2.203e-02, valid_loss 5.138e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 260 : forward loss 2.603e-04, backward loss 4.034e-02, reinference loss 1.829e-02, pi_loss 3.896e-02, valid_loss 3.230e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 280 : forward loss 2.266e-04, backward loss 3.086e-02, reinference loss 1.053e-02, pi_loss 2.292e-02, valid_loss 3.222e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 260 : forward loss 3.078e-04, backward loss 3.641e-02, reinference loss 7.334e-03, pi_loss 1.061e-02, valid_loss 4.153e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 300 : forward loss 1.842e-04, backward loss 2.501e-02, reinference loss 9.187e-03, pi_loss 3.121e-02, valid_loss 3.299e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 280 : forward loss 2.681e-04, backward loss 2.782e-02, reinference loss 8.218e-03, pi_loss 9.934e-03, valid_loss 2.806e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 320 : forward loss 1.717e-04, backward loss 2.212e-02, reinference loss 5.088e-03, pi_loss 2.427e-02, valid_loss 3.233e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 300 : forward loss 1.913e-04, backward loss 1.823e-02, reinference loss 5.309e-03, pi_loss 1.119e-02, valid_loss 2.593e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 340 : forward loss 1.448e-04, backward loss 1.416e-02, reinference loss 3.130e-03, pi_loss 2.408e-02, valid_loss 2.843e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 360 : forward loss 1.176e-04, backward loss 9.159e-03, reinference loss 2.559e-03, pi_loss 1.632e-02, valid_loss 2.171e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 320 : forward loss 1.715e-04, backward loss 1.339e-02, reinference loss 3.735e-03, pi_loss 1.279e-02, valid_loss 2.341e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 380 : forward loss 9.652e-05, backward loss 5.342e-03, reinference loss 2.424e-03, pi_loss 3.029e-02, valid_loss 2.066e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 340 : forward loss 1.429e-04, backward loss 8.198e-03, reinference loss 5.344e-03, pi_loss 2.257e-02, valid_loss 2.757e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 400 : forward loss 1.111e-04, backward loss 8.987e-03, reinference loss 3.095e-03, pi_loss 2.799e-02, valid_loss 2.284e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 360 : forward loss 1.449e-04, backward loss 7.602e-03, reinference loss 2.363e-03, pi_loss 2.524e-02, valid_loss 3.623e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 420 : forward loss 8.847e-05, backward loss 4.985e-03, reinference loss 1.847e-03, pi_loss 1.512e-02, valid_loss 2.193e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 440 : forward loss 9.589e-05, backward loss 7.308e-03, reinference loss 1.563e-03, pi_loss 1.968e-02, valid_loss 2.314e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 380 : forward loss 1.313e-04, backward loss 8.591e-03, reinference loss 2.199e-03, pi_loss 3.166e-02, valid_loss 4.135e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 460 : forward loss 7.649e-05, backward loss 3.362e-03, reinference loss 1.480e-03, pi_loss 2.909e-02, valid_loss 2.265e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 400 : forward loss 1.132e-04, backward loss 3.683e-03, reinference loss 2.644e-03, pi_loss 1.912e-02, valid_loss 3.817e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 480 : forward loss 9.988e-05, backward loss 4.966e-03, reinference loss 1.702e-03, pi_loss 4.483e-02, valid_loss 2.697e-02 [TRAIN]
CRITICAL:root:BasicDense(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=512, bias=True)
  )
)
CRITICAL:root:[Seed1] Epoch 420 : forward loss 9.710e-05, backward loss 2.632e-03, reinference loss 1.508e-03, pi_loss 9.954e-03, valid_loss 3.524e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 500 : forward loss 1.093e-04, backward loss 5.829e-03, reinference loss 3.746e-03, pi_loss 2.304e-02, valid_loss 2.343e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 520 : forward loss 1.125e-04, backward loss 7.026e-03, reinference loss 3.160e-03, pi_loss 2.646e-02, valid_loss 3.569e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 440 : forward loss 8.442e-05, backward loss 1.624e-03, reinference loss 1.057e-03, pi_loss 4.927e-03, valid_loss 2.971e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 540 : forward loss 1.073e-04, backward loss 1.109e-02, reinference loss 2.776e-03, pi_loss 5.317e-02, valid_loss 4.414e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 460 : forward loss 7.959e-05, backward loss 2.585e-03, reinference loss 1.327e-03, pi_loss 6.271e-02, valid_loss 2.163e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 560 : forward loss 9.126e-05, backward loss 6.725e-03, reinference loss 1.988e-03, pi_loss 1.004e-01, valid_loss 4.689e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 580 : forward loss 1.008e-04, backward loss 7.421e-03, reinference loss 2.461e-03, pi_loss 4.674e-02, valid_loss 4.366e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 480 : forward loss 8.138e-05, backward loss 2.074e-03, reinference loss 1.357e-03, pi_loss 3.309e-02, valid_loss 1.837e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 600 : forward loss 7.464e-05, backward loss 3.903e-03, reinference loss 2.792e-03, pi_loss 1.584e-02, valid_loss 4.331e-02 [TRAIN]
CRITICAL:root:BasicDense(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=512, bias=True)
  )
)
CRITICAL:root:[Seed1] Epoch 500 : forward loss 7.320e-05, backward loss 2.045e-03, reinference loss 1.166e-03, pi_loss 1.233e-02, valid_loss 1.766e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 620 : forward loss 6.672e-05, backward loss 1.837e-03, reinference loss 1.947e-03, pi_loss 1.121e-02, valid_loss 3.088e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 520 : forward loss 7.626e-05, backward loss 2.185e-03, reinference loss 1.016e-03, pi_loss 1.411e-02, valid_loss 1.721e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 640 : forward loss 6.265e-05, backward loss 1.554e-03, reinference loss 8.999e-04, pi_loss 1.402e-02, valid_loss 2.038e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 540 : forward loss 6.721e-05, backward loss 1.524e-03, reinference loss 8.593e-04, pi_loss 1.047e-02, valid_loss 1.779e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 660 : forward loss 6.313e-05, backward loss 2.749e-03, reinference loss 6.182e-04, pi_loss 7.343e-03, valid_loss 1.546e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 560 : forward loss 6.377e-05, backward loss 1.264e-03, reinference loss 9.725e-04, pi_loss 6.565e-03, valid_loss 1.745e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 680 : forward loss 5.725e-05, backward loss 1.347e-03, reinference loss 4.654e-04, pi_loss 6.335e-03, valid_loss 1.139e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 580 : forward loss 6.578e-05, backward loss 1.483e-03, reinference loss 1.159e-03, pi_loss 8.431e-03, valid_loss 1.398e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 700 : forward loss 5.491e-05, backward loss 9.514e-04, reinference loss 3.915e-04, pi_loss 5.485e-03, valid_loss 8.739e-03 [TRAIN]
CRITICAL:root:[Seed1] Epoch 600 : forward loss 6.191e-05, backward loss 1.107e-03, reinference loss 5.347e-04, pi_loss 1.072e-02, valid_loss 1.702e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 720 : forward loss 5.640e-05, backward loss 1.053e-03, reinference loss 5.452e-04, pi_loss 1.337e-02, valid_loss 7.961e-03 [TRAIN]
CRITICAL:root:[Seed1] Epoch 620 : forward loss 7.522e-05, backward loss 4.555e-03, reinference loss 1.449e-03, pi_loss 1.210e-02, valid_loss 1.983e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 740 : forward loss 5.036e-05, backward loss 1.125e-03, reinference loss 7.031e-04, pi_loss 7.646e-03, valid_loss 7.765e-03 [TRAIN]
CRITICAL:root:[Seed1] Epoch 640 : forward loss 6.106e-05, backward loss 1.870e-03, reinference loss 9.566e-04, pi_loss 5.613e-03, valid_loss 1.806e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 760 : forward loss 5.033e-05, backward loss 1.219e-03, reinference loss 4.563e-04, pi_loss 1.268e-02, valid_loss 8.470e-03 [TRAIN]
CRITICAL:root:[Seed1] Epoch 660 : forward loss 5.726e-05, backward loss 1.477e-03, reinference loss 8.809e-04, pi_loss 1.825e-02, valid_loss 1.896e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 780 : forward loss 5.031e-05, backward loss 1.036e-03, reinference loss 6.985e-04, pi_loss 9.227e-03, valid_loss 1.012e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 680 : forward loss 5.795e-05, backward loss 2.075e-03, reinference loss 7.084e-04, pi_loss 7.714e-03, valid_loss 2.017e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 800 : forward loss 4.765e-05, backward loss 1.090e-03, reinference loss 4.634e-04, pi_loss 5.611e-03, valid_loss 1.035e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 700 : forward loss 5.351e-05, backward loss 1.447e-03, reinference loss 8.953e-04, pi_loss 1.629e-02, valid_loss 1.658e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 820 : forward loss 4.619e-05, backward loss 8.751e-04, reinference loss 3.928e-04, pi_loss 8.214e-03, valid_loss 1.006e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 720 : forward loss 5.216e-05, backward loss 7.298e-04, reinference loss 3.908e-04, pi_loss 5.847e-03, valid_loss 1.367e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 840 : forward loss 4.908e-05, backward loss 1.178e-03, reinference loss 7.285e-04, pi_loss 3.234e-02, valid_loss 1.064e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 740 : forward loss 5.186e-05, backward loss 1.105e-03, reinference loss 8.457e-04, pi_loss 5.401e-03, valid_loss 1.406e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 860 : forward loss 5.003e-05, backward loss 3.015e-03, reinference loss 1.148e-03, pi_loss 2.977e-02, valid_loss 1.320e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 760 : forward loss 4.996e-05, backward loss 9.065e-04, reinference loss 5.109e-04, pi_loss 1.257e-02, valid_loss 1.327e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 880 : forward loss 4.741e-05, backward loss 2.497e-03, reinference loss 1.062e-03, pi_loss 5.488e-02, valid_loss 1.564e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 780 : forward loss 5.053e-05, backward loss 1.620e-03, reinference loss 1.290e-03, pi_loss 1.871e-02, valid_loss 1.431e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 900 : forward loss 6.021e-05, backward loss 3.131e-03, reinference loss 3.411e-03, pi_loss 4.815e-02, valid_loss 2.195e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 800 : forward loss 5.310e-05, backward loss 2.289e-03, reinference loss 7.272e-04, pi_loss 2.106e-02, valid_loss 1.485e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 920 : forward loss 9.915e-05, backward loss 1.424e-02, reinference loss 3.509e-03, pi_loss 7.158e-02, valid_loss 2.844e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 820 : forward loss 5.585e-05, backward loss 2.231e-03, reinference loss 1.918e-03, pi_loss 1.791e-02, valid_loss 1.672e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 940 : forward loss 7.293e-05, backward loss 6.996e-03, reinference loss 5.059e-03, pi_loss 3.160e-02, valid_loss 3.152e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 840 : forward loss 4.928e-05, backward loss 1.786e-03, reinference loss 1.528e-03, pi_loss 8.918e-03, valid_loss 1.614e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 960 : forward loss 5.787e-05, backward loss 2.610e-03, reinference loss 1.523e-03, pi_loss 1.780e-02, valid_loss 3.053e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 860 : forward loss 4.400e-05, backward loss 9.488e-04, reinference loss 4.392e-04, pi_loss 9.974e-03, valid_loss 1.569e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 980 : forward loss 4.764e-05, backward loss 2.486e-03, reinference loss 6.959e-04, pi_loss 1.488e-02, valid_loss 2.695e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 880 : forward loss 4.316e-05, backward loss 9.534e-04, reinference loss 5.213e-04, pi_loss 2.832e-02, valid_loss 1.544e-02 [TRAIN]
CRITICAL:root:BasicDense(
  (layers): ModuleList(
    (0): Linear(in_features=2, out_features=256, bias=True)
    (1): Linear(in_features=256, out_features=512, bias=True)
  )
)
CRITICAL:root:[Seed0] Epoch 1000 : forward loss 5.139e-05, backward loss 1.032e-03, reinference loss 5.090e-04, pi_loss 1.451e-02, valid_loss 2.208e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 900 : forward loss 5.622e-05, backward loss 2.150e-03, reinference loss 2.151e-03, pi_loss 1.317e-02, valid_loss 1.624e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 1020 : forward loss 5.286e-05, backward loss 2.818e-03, reinference loss 1.174e-03, pi_loss 3.491e-02, valid_loss 1.999e-02 [TRAIN]
CRITICAL:root:[Seed1] Epoch 920 : forward loss 4.788e-05, backward loss 1.883e-03, reinference loss 1.219e-03, pi_loss 7.823e-03, valid_loss 1.546e-02 [TRAIN]
CRITICAL:root:[Seed0] Epoch 1040 : forward loss 5.496e-05, backward loss 2.809e-03, reinference loss 2.103e-03, pi_loss 4.863e-02, valid_loss 2.110e-02 [TRAIN]
Traceback (most recent call last):
  File "run_hybrid_doubledonut.py", line 183, in <module>
    pool.map(do_all(), range(start_seed, start_seed+n_seeds))
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 364, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 765, in get
    self.wait(timeout)
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 762, in wait
    self._event.wait(timeout)
  File "/home/fanthomme/miniconda3/lib/python3.8/threading.py", line 558, in wait
    signaled = self._cond.wait(timeout)
  File "/home/fanthomme/miniconda3/lib/python3.8/threading.py", line 302, in wait
    waiter.acquire()
KeyboardInterrupt
Process SpawnPoolWorker-1:
Traceback (most recent call last):
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
  File "/home/fanthomme/riley/SensorFusion/run_hybrid_doubledonut.py", line 165, in __call__
    trainer.train()
  File "/home/fanthomme/riley/SensorFusion/trainer.py", line 488, in train
    fwd_loss, bkw_loss, reinf_loss = self.tuple_loss(net, env)
  File "/home/fanthomme/riley/SensorFusion/losses.py", line 23, in __call__
    rooms, positions, actions = env.static_replay(actions)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1737, in static_replay
    room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1701, in __replay_one_traj
    obs, reward, end_traj, info = self.step(action)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 2034, in step
    obs = self.get_observation(deepcopy(new_room), deepcopy(rectified_new_pos), action=deepcopy(rectified_action))
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1623, in get_observation
    image_batch = self.retina.activity(position, color)
  File "/home/fanthomme/riley/SensorFusion/retina.py", line 89, in activity
    tmp = self.base_retina.activity(positions_batch[:, obj_idx, :])
  File "/home/fanthomme/riley/SensorFusion/retina.py", line 54, in activity
    A_ = (self.A / tch.sqrt(2*np.pi*self.sigma_p)).unsqueeze(0)
KeyboardInterrupt
Process SpawnPoolWorker-2:
Traceback (most recent call last):
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/home/fanthomme/miniconda3/lib/python3.8/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
  File "/home/fanthomme/riley/SensorFusion/run_hybrid_doubledonut.py", line 165, in __call__
    trainer.train()
  File "/home/fanthomme/riley/SensorFusion/trainer.py", line 488, in train
    fwd_loss, bkw_loss, reinf_loss = self.tuple_loss(net, env)
  File "/home/fanthomme/riley/SensorFusion/losses.py", line 23, in __call__
    rooms, positions, actions = env.static_replay(actions)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1737, in static_replay
    room, pos, act =  self.__replay_one_traj(actions_batch_local[b], start_room=None, start_pos=None)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1681, in __replay_one_traj
    self.reset()
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 2011, in reset
    obs = self.get_observation(self.agent_room, self.agent_position, is_reset=True)
  File "/home/fanthomme/riley/SensorFusion/environment.py", line 1623, in get_observation
    image_batch = self.retina.activity(position, color)
  File "/home/fanthomme/riley/SensorFusion/retina.py", line 89, in activity
    tmp = self.base_retina.activity(positions_batch[:, obj_idx, :])
  File "/home/fanthomme/riley/SensorFusion/retina.py", line 48, in activity
    r = r.float().to(self.device)
KeyboardInterrupt
