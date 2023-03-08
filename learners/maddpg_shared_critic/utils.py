import numpy as np
import torch 
import torch.nn as nn
from copy import copy

class OUActionNoiseGPU(object):
    def __init__(self, mu: torch.tensor, sigma :float=0.15, theta :float=0.2, dt :float=1e-2, x0 :torch.tensor=None):
        self.theta = theta
        self.mu = mu 
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * torch.normal(mean=0, std=1, size=self.mu.shape).cuda()
        self.x_prev = copy(x)
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu.cpu().detach(), self.sigma
        )

class ReplayBuffer(object):
    """
    Needs to be modified to account for the number of agents.
    It will receive a batch of states, actions, rewards, next_states, and done
    The batch is the number of agents

    observations = [500, 62]
    
    """
    def __init__(self, max_size, input_shape, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size,  *input_shape), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size, n_actions), dtype=torch.float32)
        self.reward_memory = torch.zeros((self.mem_size,1), dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.float32)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index:index+self.n_agents] = state
        self.new_state_memory[index:index+self.n_agents] = state_
        self.action_memory[index:index+self.n_agents] = action
        self.reward_memory[index:index+self.n_agents] = reward
        self.terminal_memory[index:index+self.n_agents] = 1 - done
        self.mem_cntr += self.n_agents

    def store_single_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index:index+self.n_agents] = state
        self.new_state_memory[index:index+self.n_agents] = state_
        self.action_memory[index:index+self.n_agents] = action
        self.reward_memory[index:index+self.n_agents] = reward
        self.terminal_memory[index:index+self.n_agents] = 1 - done
        self.mem_cntr += self.n_agents
        
    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states.cuda(), actions.cuda(), rewards.cuda(), states_.cuda(), terminal.cuda()

def test():
    rb = ReplayBuffer(100, (4,), 1)
    for i in range(100):
                            # state (4,),            action (1,),      reward ,         next_state(4,),   done
        rb.store_transition(torch.tensor([i]), torch.tensor([i]), torch.tensor([i]), torch.tensor([i]), 0)

    sample = rb.sample_buffer(10)
    print(sample[0].shape)
    print(sample[0].device)


    mu = torch.tensor([0.0, 1.0]).cuda()
    noise = OUActionNoiseGPU(mu)
    print(noise().device)
    print(noise())

if __name__ == "__main__":
    test()