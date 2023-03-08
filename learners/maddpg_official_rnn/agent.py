import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import OrnsteinUhlenbeckProcess


from net import Actor, Critic

class Agent:
    def __init__(self, env, n_agent, noise, args, actor_lr=3e-3, critic_lr=3e-3, gamma=0.99, tau=0.001):
        
        self.tau = tau
        self.gamma = gamma
        self.noise = noise
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Noise comes from OrnsteinUhlenbeckProcess
        self.actor_dims = env.observation_space[1][n_agent].shape[0]
        self.critic_dims = env.observation_space[0].shape[0] * env.observation_space[0].shape[1]
        self.n_actions = env.action_space[n_agent].shape[0]
        self.index = n_agent
        self.agent_name = "agent_number_{}".format(n_agent)
        
        
        self.actor = Actor(self.actor_dims, self.n_actions, chkpt_dir=args.save_dir, name=self.agent_name + "_actor")
        self.critic = Critic(self.critic_dims, self.n_actions * env.num_particles, chkpt_dir=args.save_dir, name=self.agent_name + "_critic")
        self.target_actor = Actor(self.actor_dims, self.n_actions, chkpt_dir=args.save_dir, name=self.agent_name + "_target_actor")
        self.target_critic = Critic(self.critic_dims, self.n_actions * env.num_particles, chkpt_dir=args.save_dir, name=self.agent_name + "_target_critic")
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.target_actor_optimizer = torch.optim.Adam(self.target_actor.parameters(), lr=self.actor_lr)
        self.target_critic_optimizer = torch.optim.Adam(self.target_critic.parameters(), lr=self.critic_lr)
        
        self.target_actor.hard_update(self.actor)
        self.target_critic.hard_update(self.critic)

        
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        self.target_actor.soft_update(self.actor, tau)
        self.target_critic.soft_update(self.critic, tau)

    def _preprocesObservationTraining(self, obs):
        """
        Observation: Matrix (num_neig + 1,)
        """
        return obs[self.index]

    def choose_action(self, observation, hidden, test=False):
        """
        Assumes that everything is on cuda already
        """
        observation = self._preprocesObservationTraining(observation).cuda()
        with torch.no_grad():
            mu, hidden_next = self.actor.forward(observation, hidden)
            if not test:
                mu_prime = mu + self.noise.sample()
            else:
                mu_prime = mu
        return mu_prime.detach().squeeze(0), hidden_next
    
    def save(self, path_save):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def load_single_checkpoint(self, checkpoint):
        params = torch.load(checkpoint)
        self.actor.load_state_dict(params)
        
