
import sys
sys.path.append(r"C:\Users\victo\Desktop\VU master\Multiagent\multi-rl-crowd-sim-main")

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

print("imports [OK]")


class Agent(object):
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        checkpoint_dir,
        checkpoint_best,
        index,
        replay_buffer,
        noise,
        recurrent=False,
        gamma=0.99,
        n_actions=2,
        layer1_size=32,
        layer2_size=32,
        batch_size=64,
        update_rate=3,

    ):
        self.recurrent = recurrent
        if recurrent:
            from maddpg.models.ddpg_RNN.DDPG_network import ActorNetwork, CriticNetwork
            from maddpg.models.ddpg_RNN.utils import OUActionNoiseGPU

        else:
            from maddpg.models.DDPG.DDPG_network import ActorNetwork, CriticNetwork
            from maddpg.models.DDPG.utils import  OUActionNoiseGPU

        self.index = index
        self.gamma = gamma
        self.tau = tau
        self.memory = replay_buffer
        self.batch_size = batch_size
        self.actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="Actor",
            chkpt_dir=checkpoint_dir,
            chkpt_best_dir=checkpoint_best
        )
        self.critic = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="Critic",
            chkpt_dir=checkpoint_dir,
            chkpt_best_dir=checkpoint_best

        )

        self.target_actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="TargetActor",
            chkpt_dir=checkpoint_dir,
            chkpt_best_dir=checkpoint_best
        )
        self.target_critic = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="TargetCritic",
            chkpt_dir=checkpoint_dir,
            chkpt_best_dir=checkpoint_best

        )

        self.noise = noise

        self.update_network_parameters(tau=1)
        self.count = 0
        self.update_rate = update_rate
        # self.target = torch.zeros((self.batch_size, 1)).cuda()
        
    def _preprocesObservationTraining(self, obs):
        """
        Observation: Matrix (num_neig + 1,)
        """
        return obs[self.index]

    def choose_action(self, observation):
        """
        Observation: Matrix (num_Drones + 3,)
        Assumes that everything is on cuda already
        """
        self.actor.eval()
        if not self.recurrent:
            observation = torch.flatten(self._preprocesObservationTraining(observation)).cuda()
        else:
            observation = self._preprocesObservationTraining(observation).cuda()
        with torch.no_grad():
            mu = self.actor.forward(observation)
            mu_prime = mu + self.noise()

        self.actor.train()
        return mu_prime.detach()

    def remember(self, state, action, reward, new_state, done):
        if not self.recurrent:
            state = torch.flatten(state)
            new_state =  torch.flatten(new_state)
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 0, 0, False
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        self.target_actor.eval()
        self.target_critic.eval()
        # self.critic.eval()
        with torch.no_grad():
            target_actions = self.target_actor.forward(new_state)
            critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action) 
        reward = reward.view(self.batch_size, 1)
        target = reward + self.gamma * critic_value_ * done.reshape(-1,1)
        # target = []
        # for j in range(self.batch_size):
        #     target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        # target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        if self.count % self.update_rate == 0:
            self.update_network_parameters()
        self.count += 1
        return actor_loss, critic_loss, True


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def save_models_best(self):
        self.actor.save_checkpoint_best()
        self.target_actor.save_checkpoint_best()
        self.critic.save_checkpoint_best()
        self.target_critic.save_checkpoint_best()
    
    def load_models(self, silent=False):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print("Checking Actor parameters")

        for param in current_actor_dict:
            print(param, torch.equal(original_actor_dict[param], current_actor_dict[param]))
        print("Checking critic parameters")
        for param in current_critic_dict:
            print(
                param, torch.equal(original_critic_dict[param], current_critic_dict[param])
            )
        input()