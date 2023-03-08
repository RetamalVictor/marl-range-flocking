import os
import torch
import collections
import numpy as np
import json


class ReplayBufferMaddpg(object):
    """
    Could be vectorized with torch
    """
    def __init__(self, env, buffer_capacity=1_000_000, batch_size=128, chunk_size=10, min_size_buffer=8_000):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.chunk_size = chunk_size
        self.buffer_counter = 0
        self.n_games = 0
        self.n_agents = env.num_particles
        self.list_actors_dimension = [env.observation_space[1][index].shape[0] for index in range(self.n_agents)]
        self.critic_dimension = env.observation_space[0].shape     
        self.list_actor_n_actions = [env.action_space[index].shape[0] for index in range(self.n_agents)]
        
        # Critic shape is [buffer, chunck, critic_dimension] and not including the agents bc is the same for all agents
        self.states = torch.zeros((self.buffer_capacity, *self.critic_dimension)).cuda()
        self.rewards = torch.zeros((self.buffer_capacity, self.n_agents, 1)).cuda()
        self.next_states = torch.zeros((self.buffer_capacity, *self.critic_dimension)).cuda()
        self.dones = torch.zeros((self.buffer_capacity, self.n_agents, 1)).cuda()

        self.list_actors_states = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actors_dimension[0])).cuda()
        self.list_actors_next_states = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actors_dimension[0])).cuda()
        self.list_actors_actions = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actor_n_actions[0])).cuda()
        
        self.states_sample = torch.zeros((self.batch_size, self.chunk_size, *self.critic_dimension)).cuda()
        self.rewards_sample = torch.zeros((self.batch_size, self.chunk_size, self.n_agents, 1)).cuda()
        self.next_states_sample = torch.zeros((self.batch_size, self.chunk_size, *self.critic_dimension)).cuda()
        self.dones_sample = torch.zeros((self.batch_size, self.chunk_size, self.n_agents, 1)).cuda()

        self.list_actors_states_sample = torch.zeros((self.n_agents, self.batch_size, self.chunk_size, self.list_actors_dimension[0])).cuda()
        self.list_actors_next_states_sample = torch.zeros((self.n_agents, self.batch_size, self.chunk_size, self.list_actors_dimension[0])).cuda()
        self.list_actors_actions_sample = torch.zeros((self.n_agents, self.batch_size, self.chunk_size, self.list_actor_n_actions[0])).cuda()
        
    
    def __len__(self):
        return self.buffer_counter
        
    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    
    def update_n_games(self):
        self.n_games += 1
          
    def add_record(self, actor_states, actor_next_states, actions, state, next_state, reward, done):
        
        index = self.buffer_counter % self.buffer_capacity

        # for agent_index in range(self.n_agents):
        self.list_actors_states[:, index, :] = actor_states
        self.list_actors_next_states[:, index, :] = actor_next_states
        self.list_actors_actions[:, index, :] = actions

        self.states[index] = state
        self.next_states[index] = next_state
        self.rewards[index] = reward.view(-1,1)
        self.dones[index] = done.view(-1,1)
            
        self.buffer_counter += 1

    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)

        start_idx = np.random.choice(buffer_range - self.chunk_size, self.batch_size, replace=False)

        batch = 0
        for idx in start_idx:
            count = 0
            for chunck_step in range(idx, idx + self.chunk_size):
                # Take indices
                self.states_sample[batch, count] = self.states[chunck_step]
                self.rewards_sample[batch, count] = self.rewards[chunck_step]
                self.next_states_sample[batch, count] = self.next_states[chunck_step]
                self.dones_sample[batch, count] = self.dones[chunck_step]

                self.list_actors_states_sample[:, batch, count] = self.list_actors_states[:, chunck_step, :]
                self.list_actors_next_states_sample[:, batch, count] = self.list_actors_next_states[:, chunck_step, :]
                self.list_actors_actions_sample[:, batch, count] = self.list_actors_actions[:, chunck_step, :] 
                count += 1
            batch += 1

        return self.states_sample.clone() \
            , self.rewards_sample.clone() \
            , self.next_states_sample.clone() \
            , self.dones_sample.clone() \
            , self.list_actors_states_sample.clone() \
            , self.list_actors_next_states_sample.clone() \
            , self.list_actors_actions_sample.clone()



    
    def save(self, folder_path):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        
        np.save(folder_path + '/states.npy', self.states.cpu().numpy())
        np.save(folder_path + '/rewards.npy', self.rewards.cpu().numpy())
        np.save(folder_path + '/next_states.npy', self.next_states.cpu().numpy())
        np.save(folder_path + '/dones.npy', self.dones.cpu().numpy())
        
        for index in range(self.n_agents):
            np.save(folder_path + '/states_actor_{}.npy'.format(index), self.list_actors_states[index].cpu().numpy())
            np.save(folder_path + '/next_states_actor_{}.npy'.format(index), self.list_actors_next_states[index].cpu().numpy())
            np.save(folder_path + '/actions_actor_{}.npy'.format(index), self.list_actors_actions[index].cpu().numpy())
            
        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}
        
        with open(folder_path + '/dict_info.json', 'w') as f:
            json.dump(dict_info, f)
            
    def load(self, folder_path):
        self.states = np.load(folder_path + '/states.npy')
        self.rewards = np.load(folder_path + '/rewards.npy')
        self.next_states = np.load(folder_path + '/next_states.npy')
        self.dones = np.load(folder_path + '/dones.npy')
        
        self.list_actors_states = [np.load(folder_path + '/states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_next_states = [np.load(folder_path + '/next_states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_actions = [np.load(folder_path + '/actions_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        
        with open(folder_path + '/dict_info.json', 'r') as f:
            dict_info = json.load(f)
        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]

#######################################################################################################################
