import numpy as np 
import torch

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.sample_sigma = sigma
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sample_sigma * np.sqrt(self.dt) * torch.randn(size=(self.size,)).cuda()
        self.x_prev = x
        # self.n_steps += 1
        return x

    def update_sigma(self):
        self.sample_sigma = self.current_sigma
        self.n_steps += 1


    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros((self.size)).cuda()


def test_ou():
    import matplotlib.pyplot as plt
    ou = OrnsteinUhlenbeckProcess(0.01, sigma=0.2,sigma_min=0.001)
    # plt.plot([ou.current_sigma for _ in range(1000)])
    sigmas = []
    sample = []
    for i in range(100):
        sample.append(ou.sample().cpu().numpy())
        sigmas.append(ou.sample_sigma)
    for i in range(2000):
        sample.append(ou.sample().cpu().numpy())
        sample.append(ou.sample().cpu().numpy())
        # print(ou.sample())
        if i % 100 == 0:
            ou.reset_states()
        ou.update_sigma()
        sigmas.append(ou.sample_sigma)
    plt.plot(sigmas)
    plt.plot(sample)
    plt.show()

if __name__ == "__main__":
    test_ou()