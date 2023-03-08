import os, time, torch, gym, random

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from agent import Agent
from memory import ReplayBufferMaddpg
from utils import OrnsteinUhlenbeckProcess

torch.autograd.set_detect_anomaly(True)

class SuperAgent:
    def __init__(self, args,  env:gym.Env, path_save:str= "/tmp", path_load:str = "/tmp" ):
        self.path_save = path_save
        self.path_load = path_load
        self.replay_buffer = ReplayBufferMaddpg(env)
        self.random_process = OrnsteinUhlenbeckProcess(size=env.action_space[0].shape[0], theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)
        self.n_agents = env.num_particles
        self.agents = [
            Agent(env=env, n_agent=agent, noise=self.random_process, args=args) for agent in range(self.n_agents)
            ]
        
    def get_actions(self, actor_states, test=False):
        list_actions = [self.agents[index].choose_action(actor_states, test) for index in range(self.n_agents)]
        return torch.stack(list_actions)    
      

    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        full_path = f"{self.path_save}/save_agent_{date_now}"
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        
        for agent in self.agents:
            agent.save(full_path)
            
        self.replay_buffer.save(full_path)
    
    def load(self):
        # full_path = self.path_load
        for agent in self.agents:
            agent.load()
            
        # self.replay_buffer.load(full_path)

    def load_replay_buffer(self):
        full_path = self.path_load
        self.replay_buffer.load(full_path)

    def load_single_checkpoint(self, path):
        print("Loading checkpoint from: ", path)
        for agent in self.agents:
            agent.load_single_checkpoint(path)

    def load_scaled_checkpoint(self, ch_path, total=5):
        """
        Path is a dir containing the checkpoints of the agents
        
        """
        print("Loading checkpoint from: ", ch_path)
        for agent in self.agents:
            scaled_path = os.path.join(ch_path, f"agent_number_{random.randint(0, total)}_actor_ddpg.pt")
            agent.load_single_checkpoint(scaled_path)
        print(f"Loaded scaled checkpoint with {total} checkpoints")

    def train(self):
        if self.replay_buffer.check_buffer_size() == False:
            return
        
        # Sample a batch from the replay buffer
        states, rewards, next_states, done, actors_states, actors_next_states, actors_action = self.replay_buffer.get_minibatch()
        states = states.reshape(-1, self.n_agents * 9) # Hardcoded
        next_states = next_states.reshape(-1, self.n_agents * 9)
        concat_actors_action = actors_action.reshape(-1, self.n_agents * 2)

        # Compute the target actions and the policy actions
        with torch.no_grad():
            target_actions = [self.agents[index].target_actor(actors_next_states[index]) for index in range(self.n_agents)]
            concat_target_actions = torch.cat(target_actions, dim=1)
        
            # Compute the target critic values and the critic values
            target_critic_values = [self.agents[index].target_critic(
                (next_states,
                concat_target_actions)) for index in range(self.n_agents)]
                
        critic_values = [self.agents[index].critic((states, concat_actors_action)) for index in range(self.n_agents)]
        
        # Compute the targets Q values
        targets = [rewards[:, index] + self.agents[index].gamma * target_critic_values[index] * (1-done[:, index]) for index in range(self.n_agents)]
        
        # Compute the losses
        critic_losses = [F.mse_loss(targets[index], critic_values[index]) for index in range(self.n_agents)]
        policy_actions = [self.agents[index].actor(actors_states[index]) for index in range(self.n_agents)]
        concat_policy_actions = torch.cat(policy_actions, dim=1)
        
        actor_losses = [-self.agents[index].critic((states.detach(), concat_policy_actions.detach())) for index in range(self.n_agents)]
        actor_losses = [torch.mean(actor_losses[index]) for index in range(self.n_agents)]

        # Update the networks
        for index in range(self.n_agents):
            self.agents[index].critic_optimizer.zero_grad()
            self.agents[index].actor_optimizer.zero_grad()
            critic_losses[index].backward(retain_graph=True)
            actor_losses[index].backward(retain_graph=True)
            self.agents[index].critic_optimizer.step()
            self.agents[index].actor_optimizer.step()
            self.agents[index].update_target_networks()
