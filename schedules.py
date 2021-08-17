from collections import defaultdict
# Order of param_groups :  gating, representation, z_encoding, backward, forward
# NOTE: the first time schedulers are called is always with step=-1

class DefaultSchedulerLR:
    def __init__(self, **kwargs):
        # self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4, 1e-4]
        self.current_lr = [1e-3, 1e-3, 0., 1e-3, 1e-4]

    def get_current_lr(self, step):
        return self.current_lr, False

class DefaultSchedulerWeights:
    def __init__(self, **kwargs):
        self.current_weights = [10, 10, 0., 1, 0, 0]

    def get_current_weights(self, step):
        return self.current_weights, False


class DefaultNoFBSchedulerLR:
    def __init__(self, **kwargs):
        # self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4, 1e-4]
        self.current_lr = [0., 0., 0., 1e-3, 1e-4]

    def get_current_lr(self, step):
        return self.current_lr, False

class DefaultNoFBSchedulerWeights:
    def __init__(self, **kwargs):
        self.current_weights = [0, 0, 0., 1, 0, 0]

    def get_current_weights(self, step):
        return self.current_weights, False


class OffshelfDefaultSchedulerLR:
    def __init__(self, **kwargs):
        # self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4, 1e-4]
        # recurrence, representation, z_encoder, decoder
        self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4]

    def get_current_lr(self, step):
        return self.current_lr, False


class OffshelfDontTouchEncodingSchedulerLR:
    def __init__(self, **kwargs):
        # self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4, 1e-4]
        # recurrence, representation, z_encoder, decoder
        self.current_lr = [5e-4, 0., 0., 5e-4]

    def get_current_lr(self, step):
        return self.current_lr, False


class SchedulerWeightsForNoisyDoubleDonut:
    def __init__(self, **kwargs):
        # self.current_weights = [100, 100, 100, 1, 0, 0]
        self.current_weights = [100, 100, 0., 1, 0, 0]

    def get_current_weights(self, step):
        return self.current_weights, False

class SchedulerLRForNoisyDoubleDonut:
    def __init__(self, **kwargs):
        # self.current_lr = [5e-4, 5e-4, 5e-4, 5e-4, 1e-4]
        self.current_lr = [5e-4, 5e-4, 0., 5e-4, 1e-4]

    def get_current_lr(self, step):
        return self.current_lr, False

class AnnealedSchedulerLR:
    def __init__(self, **kwargs):
        self.current_lr = [5e-4, 1e-3, 0., 1e-3, 1e-3,]
        self.switch_at = [2000, 4000, 6000, 2**20]
        self.idx = 0

        self.lr_to_switch_to = [
            [5e-4, 5e-4, 0., 5e-4, 5e-4],
            [1e-4, 5e-4, 0., 5e-4, 5e-4],
            [1e-4, 1e-4, 0., 1e-4, 0],
            None
        ]

    def get_current_lr(self, step):
        if step != self.switch_at[self.idx]:
            return self.current_lr, False
        else:
            self.current_lr = self.lr_to_switch_to[self.idx]
            self.idx = self.idx +  1
            return self.current_lr, True

class AnnealedSchedulerWeights:
    def __init__(self, **kwargs):
        self.current_weights = [10, 10, 0., 1, 0, 0]
        self.switch_at = [2**20]
        self.idx = 0
        self.weights_to_switch_to = [None]

    def get_current_weights(self, step):
        if step != self.switch_at[self.idx]:
            return self.current_weights, False
        else:
            self.current_weights = self.weights_to_switch_to[self.idx]
            self.idx = self.idx +  1
            return self.current_weights, True


class AlternatingSchedulerLR:
    def __init__(self, **kwargs):
        self.current_lr = [5e-4, 1e-3, 0., 1e-3, 1e-3,]
        self.switch_at = [4000, 8000, 2**20]
        self.idx = 0


        self.lr_to_switch_to = [
            [1e-4, 5e-4, 0., 5e-4, 5e-4],
            [1e-4, 1e-4, 0., 1e-4, 0],
            None
        ]

    def get_current_lr(self, step):
        if step != self.switch_at[self.idx]:
            return self.current_lr, False
        else:
            self.current_lr = self.lr_to_switch_to[self.idx]
            self.idx = self.idx +  1
            return self.current_lr, True

class AlternatingSchedulerWeights:
    def __init__(self, **kwargs):
        self.current_weights = [1, 1, 0., 0, 0, 0]
        self.switch_at = [2000, 4000, 6000, 8000, 2**20]
        self.idx = 0
        self.weights_to_switch_to = [
        [0, 0, 0, 1, 0, 0],
        [1, 1, 0., 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        None
        ]

    def get_current_weights(self, step):
        if step != self.switch_at[self.idx]:
            return self.current_weights, False
        else:
            self.current_weights = self.weights_to_switch_to[self.idx]
            self.idx = self.idx +  1
            return self.current_weights, True

lr_scheduler_list = {
                    'default': DefaultSchedulerLR,
                    'default_no_fb': DefaultNoFBSchedulerLR,
                    'annealed': AnnealedSchedulerLR,
                    'alternating': AlternatingSchedulerLR,
                    'noisy_double_donut': SchedulerLRForNoisyDoubleDonut,
                    'offshelf_default': OffshelfDefaultSchedulerLR,
                    'offshelf_dont_touch_encoding': OffshelfDontTouchEncodingSchedulerLR,
                    }

weights_scheduler_list = {
                         'default': DefaultSchedulerWeights,
                         'default_no_fb': DefaultNoFBSchedulerWeights,
                         'annealed': AnnealedSchedulerWeights,
                         'alternating': AlternatingSchedulerWeights,
                         'noisy_double_donut': SchedulerWeightsForNoisyDoubleDonut,
                         }
