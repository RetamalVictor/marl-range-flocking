import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, input_dim, nb_actions, hidden1=400, hidden2=300, init_w=3e-3,
        name="actor",
        chkpt_dir=r"tmp\ddpg",
        chkpt_best_dir=r"tmp\ddpg_best"):
        super(Actor, self).__init__()

        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg.pt")
        self.checkpoint_best_file = os.path.join(chkpt_best_dir, name + "_ddpg.pt")

        if os.path.isdir(chkpt_dir) == False:
            os.mkdir(chkpt_dir)

        self.input_dim = input_dim
        self.nb_actions = nb_actions

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
        self.to_cuda()
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out
    
    def soft_update(self, source:nn.Module, tau):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, source: nn.Module):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def save_checkpoint_best(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_best_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    def to_cuda(self):
        self.cuda()

class Critic(nn.Module):
    def __init__(self, input_dim, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, 
        name="critic",
        chkpt_dir=r"tmp\ddpg",
        chkpt_best_dir=r"tmp\ddpg_best"):
        super(Critic, self).__init__()

        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg.pt")
        self.checkpoint_best_file = os.path.join(chkpt_best_dir, name + "_ddpg.pt")
        
        if os.path.isdir(chkpt_dir) == False:
            os.mkdir(chkpt_dir)
            
        self.input_dim = input_dim
        self.nb_actions = nb_actions

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        self.to_cuda()
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,a],1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def soft_update(self, source:nn.Module, tau):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, source: nn.Module):
        for target_param, param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    def save_checkpoint_best(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_best_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    def to_cuda(self):
        self.cuda()