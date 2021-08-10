import torch as tch
import numpy as np

class OneObjectGrayRetina:
    def __init__(self, n=64**2, bounds=[-.5,.5], widths=[.4, .5], device_name='cuda'):
        self.n = n
        self.sqrt_n = int(np.sqrt(n))
        assert self.sqrt_n ** 2 == self.n
        self.bounds = bounds
        self.device = tch.device(device_name)
        self.widths = widths

        # Assume regular arrangement for now, likely not important
        x = np.linspace(*self.bounds, self.sqrt_n)
        y = np.linspace(*self.bounds, self.sqrt_n)
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        self.centers = tch.from_numpy(xy).float().to(self.device)

        # Widths also identical
        self.sigma_p = tch.from_numpy(np.array([self.widths[0] for _ in range(self.n)])).float().to(self.device)
        self.sigma_m = tch.from_numpy(np.array([self.widths[1] for _ in range(self.n)])).float().to(self.device)

        # Only on-center cells in this retina
        self.A =  tch.from_numpy(np.array([1. for _ in range(self.n)])).float().to(self.device)
        self.B = -self.A

        self.__set_A()

    def __set_A(self):
        # Chosen so that mean square activity in a box of radius 1 around an object is 1.
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        act = self.activity(tch.from_numpy(xy).to(self.device))
        self.A /= tch.sqrt((act**2).mean(dim=1)).mean()
        self.B /= tch.sqrt((act**2).mean(dim=1)).mean()


    def activity(self, r):
        # Input: batch of positions (relative to retina center) (bs, 2)
        # Output: batch of activations (ordered by the xy tiling above) (bs, self.n)
        try:
            assert r.dim() == 2
        except AttributeError:
            r = tch.from_numpy(r)
        except AssertionError:
            raise RuntimeError('Invalid number of dimensions for input ({}: {})'.format(r.dim(), r.shape))
        r = r.float().to(self.device)

        # Efficient batch vector differences https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451/4
        vect_dist = self.centers.unsqueeze(0) - r.unsqueeze(1) # c is (n, 2); r is (bs, 2), vect_dist is (bs,n,2) with b,i-th element r[b]-c[i]
        sq_dist = (vect_dist ** 2).sum(dim=-1)

        A_ = (self.A / tch.sqrt(2*np.pi*self.sigma_p)).unsqueeze(0)
        B_ = (self.B / tch.sqrt(2*np.pi*self.sigma_m)).unsqueeze(0)

        gauss_p = tch.exp(-sq_dist/(2*self.sigma_p**2).unsqueeze(0))
        gauss_m = tch.exp(-sq_dist/(2*self.sigma_m**2).unsqueeze(0))

        return A_ * gauss_p + B_ * gauss_m


class Retina:
    '''
    This is the full fledged, many objects, rgb retina
    '''
    def __init__(self, n=64**2, bounds=[-.5,.5], widths=[.3, .5], device_name='cuda'):
        self.base_retina = OneObjectGrayRetina(n=n, bounds=bounds, widths=widths, device_name=device_name)
        self.n = self.base_retina.n
        self.sqrt_n = self.base_retina.sqrt_n
        self.widths = self.base_retina.widths
        self.device = self.base_retina.device

    def activity(self, positions_batch, object_colors):
        # Positions_batch should have shape: (bs, n_objects, 2)
        # object_colors should have shape (n_objects, 3)
        # NOTE: if doing batches where the number of objects is not constant, just
        # use the max and put a "zero" object color if an object does not exist
        if len(positions_batch.shape) == 2:
            positions_batch = positions_batch.unsqueeze(0)
            object_colors = object_colors.unsqueeze(0)

        bs = positions_batch.shape[0]
        n_objects = positions_batch.shape[1]

        # OPTIMIZE: find a way to parallelize
        act = tch.zeros(bs, self.n, 3).to(self.device)
        for obj_idx in range(n_objects):
            tmp = self.base_retina.activity(positions_batch[:, obj_idx, :])
            act += tmp.view(bs, self.n, 1).repeat(1,1,3) * (object_colors[:, obj_idx].view(bs, 1, 3).repeat(1,self.n,1))
        return act
