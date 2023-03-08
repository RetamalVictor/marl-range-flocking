import os, time, torch, gym, random

from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from agent import Agent
from memory_rnn import ReplayBufferMaddpg
from utils import OrnsteinUhlenbeckProcess

torch.autograd.set_detect_anomaly(True)

class SuperAgent:
    def __init__(self, args,  env:gym.Env, path_save:str= "/tmp", path_load:str = "/tmp" ):
        self.path_save = path_save
        self.path_load = path_load
        self.replay_buffer = ReplayBufferMaddpg(env, buffer_capacity = args.buffer_size, batch_size = args.batch_size, min_size_buffer=args.min_size_buffer)
        self.random_process = OrnsteinUhlenbeckProcess(size=env.action_space[0].shape[0], theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma, sigma_min=args.ou_sigma_min)
        self.n_agents = env.num_particles
        self.agents = [
            Agent(env=env, n_agent=agent, noise=self.random_process, args=args) for agent in range(self.n_agents)
            ]
    
    def init_hidden(self):
        return [agent.actor.init_hidden() for agent in self.agents]


    def get_actions(self, actor_states, hidden_states, test=False):
        list_actions = [self.agents[index].choose_action(actor_states, hidden_states[index], test) for index in range(self.n_agents)]
        hidden_states = list(zip(*list_actions))[1]
        list_actions = list(zip(*list_actions))[0]
        return torch.stack(list_actions), hidden_states
    
    def reset_random_process(self):
        self.random_process.reset_states()

    def update_random_process(self):
        self.random_process.update_sigma()

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

    def train(self,batch_size=128,  chunk_size=10):
        if self.replay_buffer.check_buffer_size() == False:
            return
        
        # Sample a batch from the replay buffer. In the RNN case, the states are already concatenated
        states, rewards, next_states, done, actors_states, actors_next_states, actors_action = self.replay_buffer.get_minibatch()
        states = states.reshape(batch_size, chunk_size, self.n_agents * 4) # Hardcoded
        next_states = next_states.reshape(batch_size, chunk_size, self.n_agents * 4)
        concat_actors_action = actors_action.reshape(batch_size, chunk_size, self.n_agents * 2)
        
        hidden_list_target_actor = [self.agents[index].target_actor.init_hidden(batch_size) for index in range(self.n_agents)]
        hidden_list_target_critic = [self.agents[index].target_critic.init_hidden(batch_size) for index in range(self.n_agents)]
        hidden_list_critic = [self.agents[index].critic.init_hidden(batch_size) for index in range(self.n_agents)]
        hidden_list_actor = [self.agents[index].actor.init_hidden(batch_size) for index in range(self.n_agents)]
        # total_actor_loss = [0 for index in range(self.n_agents)]
        for step_i in range(chunk_size):
            # Compute the target actions and the policy actions
            with torch.no_grad():
                target_actions = [self.agents[index].target_actor(actors_next_states[index, :, step_i], hidden_list_target_actor[index]) for index in range(self.n_agents)]
                concat_target_actions = torch.cat(list(zip(*target_actions))[0], dim=1)
                hidden_list_target_actor = list(zip(*target_actions))[1]

                # Compute the target critic values and the critic values
                target_critic_values = [self.agents[index].target_critic(
                    (next_states[:, step_i],
                    concat_target_actions), hidden_list_target_critic[index]) for index in range(self.n_agents)]
                hidden_list_target_critic = list(zip(*target_critic_values))[1]
                target_critic_values = list(zip(*target_critic_values))[0]
                    
            critic_values = [self.agents[index].critic((states[:, step_i], concat_actors_action[:, step_i]), hidden_list_critic[index]) for index in range(self.n_agents)]
            hidden_list_critic = list(zip(*critic_values))[1]
            critic_values = list(zip(*critic_values))[0]
            
            # Compute the losses
            policy_actions = [self.agents[index].actor(actors_states[index, : , step_i], hidden_list_actor[index]) for index in range(self.n_agents)]
            hidden_list_actor = list(zip(*policy_actions))[1]
            policy_actions = list(zip(*policy_actions))[0]
            concat_policy_actions = torch.cat(policy_actions, dim=1)
            
            done_mask = done[:, step_i].squeeze(-1).bool()
            if torch.any(done_mask):
                for agent_index in range(self.n_agents):
                    """
                    The logic here:
                    1. We have a done mask for each agent (done_mask[:, agent_index]) (batch_size, 1)
                    2. We have the hidden states for each agent (hidden_list_target_actor[agent_index]) (batch_size, 1, hidden_size)
                    3. We have the hidden states for each agent that are done (hidden_list_target_actor[agent_index][done_mask[:, agent_index]]) (batch_size, hidden_size)
                    4. We have the hidden states for each agent that are done and we want to reset them 
                            (hidden_list_target_actor[agent_index][done_mask[:, agent_index]] = self.agents[agent_index].target_actor.init_hidden(len(hidden_list_target_actor[agent_index][done_mask[:, agent_index]]))) 
                            (batch_size, hidden_size)
                    """
                    hidden_list_target_actor[agent_index][done_mask[:, agent_index]] = self.agents[agent_index].target_actor.init_hidden(len(hidden_list_target_actor[agent_index][done_mask[:, agent_index]]))
                    hidden_list_target_critic[agent_index][done_mask[:, agent_index]] = self.agents[agent_index].target_critic.init_hidden(len(hidden_list_target_critic[agent_index][done_mask[:, agent_index]]))
                    hidden_list_critic[agent_index][done_mask[:, agent_index]] = self.agents[agent_index].critic.init_hidden(len(hidden_list_critic[agent_index][done_mask[:, agent_index]]))
                    hidden_list_actor[agent_index][done_mask[:, agent_index]] = self.agents[agent_index].actor.init_hidden(len(hidden_list_actor[agent_index][done_mask[:, agent_index]]))

        # Compute the targets Q values
        targets = [rewards[:, step_i,  index] + self.agents[index].gamma * target_critic_values[index] * (1-done[:, step_i, index]) for index in range(self.n_agents)]
        critic_losses = [F.mse_loss(targets[index], critic_values[index]) for index in range(self.n_agents)]
        actor_losses = [-self.agents[index].critic((states[:, step_i].detach(), concat_policy_actions.detach()), hidden_list_critic[index])[0] for index in range(self.n_agents)]
        # actor_losses =   list(zip(*actor_losses))[0]
        actor_losses = [torch.mean(actor_losses[index]) for index in range(self.n_agents)]
            # total_actor_loss = [total_actor_loss[index] + actor_losses[index] for index in range(self.n_agents)]

        # Update the networks
        for index in range(self.n_agents):
            self.agents[index].critic_optimizer.zero_grad()
            self.agents[index].actor_optimizer.zero_grad()
            critic_losses[index].backward(retain_graph=True)
            actor_losses[index].backward(retain_graph=True)
            self.agents[index].critic_optimizer.step()
            self.agents[index].actor_optimizer.step()
            self.agents[index].update_target_networks()
