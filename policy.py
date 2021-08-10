from numpy.random import RandomState
from scipy.signal import lfilter

class RandomPolicy:
    def __init__(self, step_size=.05, seed=0, **kwargs):
        self.seed = seed
        self.rng = RandomState(seed)
        self.step_size = step_size

    def get_batch_of_actions(self, batch_size=64, epoch_len=1):
        return self.step_size * self.rng.randn(batch_size, epoch_len, 2)

class CoherentPolicy:
    def __init__(self, step_size=.05, coherence=.6, seed=0, **kwargs):
        self.seed = seed
        self.rng = RandomState(seed)
        self.step_size = step_size
        self.coherence = 1.-coherence
        self.normalize = coherence


    def get_batch_of_actions(self, batch_size=64, epoch_len=1):
        all_actions = self.step_size * self.rng.randn(batch_size, epoch_len, 2)
        return self.normalize * lfilter([self.coherence], [1, -self.coherence], all_actions)

policy_register = {'RandomPolicy': RandomPolicy, 'CoherentPolicy': CoherentPolicy}
