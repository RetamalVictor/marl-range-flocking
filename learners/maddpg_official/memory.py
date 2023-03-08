import os
import torch
import collections
import numpy as np
import json


class ReplayBufferMaddpg(object):
    """
    Could be vectorized with torch
    """
    def __init__(self, env, buffer_capacity=1_000_000, batch_size=128, min_size_buffer=8_000):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.n_agents = env.num_particles
        self.list_actors_dimension = [env.observation_space[1][index].shape[0] for index in range(self.n_agents)]
        self.critic_dimension = env.observation_space[0].shape     
        self.list_actor_n_actions = [env.action_space[index].shape[0] for index in range(self.n_agents)]
        
        self.states = torch.zeros((self.buffer_capacity, *self.critic_dimension)).cuda()
        self.rewards = torch.zeros((self.buffer_capacity, self.n_agents, 1)).cuda()
        self.next_states = torch.zeros((self.buffer_capacity, *self.critic_dimension)).cuda()
        self.dones = torch.zeros((self.buffer_capacity, self.n_agents, 1)).cuda()

        self.list_actors_states = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actors_dimension[0])).cuda()
        self.list_actors_next_states = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actors_dimension[0])).cuda()
        self.list_actors_actions = torch.zeros((self.n_agents, self.buffer_capacity, self.list_actor_n_actions[0])).cuda()
        
        # for n in range(self.n_agents):
        #     self.list_actors_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
        #     self.list_actors_next_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
        #     self.list_actors_actions.append(np.zeros((self.buffer_capacity, self.list_actor_n_actions[n])))
    
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

        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Take indices
        state = self.states[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]

        actors_state = self.list_actors_states[:, batch_index, :]
        actors_next_state = self.list_actors_next_states[:, batch_index, :]
        actors_action = self.list_actors_actions[:, batch_index, :] 

        return state, reward, next_state, done, actors_state, actors_next_state, actors_action
        # return torch.from_numpy(state).float() \
        #     , torch.from_numpy(reward).float() \
        #     , torch.from_numpy(next_state).float() \
        #     , torch.from_numpy(done).float() \
        #     , [torch.from_numpy(actors_state[index]).float() for index in range(self.n_agents)] \
        #     , [torch.from_numpy(actors_next_state[index]).float() for index in range(self.n_agents)] \
        #     , [torch.from_numpy(actors_action[index]).float() for index in range(self.n_agents)]

    
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


class ReplayBufferSequential(object):
    def __init__(self, buffer_limit, chunk_size, n_agents, input_shape, batch_size=32):
        
        self.buffer = collections.deque(maxlen=buffer_limit)

        self.state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.new_state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.action_memory = torch.zeros((batch_size, chunk_size, n_agents), dtype=torch.float32).cuda()
        self.reward_memory = torch.zeros((batch_size,chunk_size, n_agents), dtype=torch.float32).cuda()
        self.terminal_memory = torch.zeros((batch_size, chunk_size, 1), dtype=torch.float32).cuda()

    def put(self, transition):
        """
         this transition a list of tensors.
         transition = [s, a, r, s_prime, done]
            s = [n_agents, obs_size]: obs_size = (NearestNeigbors)
            a = [n_agents]
            r = [n_agents]
            s_prime = [n_agents, obs_size]
            done = [1]
        """
        self.buffer.append(transition)


    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)

        batch = 0
        for idx in start_idx:
            count = 0
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                self.state_memory[batch, count] = s
                self.action_memory[batch, count] = a
                self.reward_memory[batch, count] = r[:,0]
                self.new_state_memory[batch, count] = s_prime
                self.terminal_memory[batch, count] = done[0]
                count += 1
            batch += 1


        return self.state_memory.clone(), \
                self.action_memory.clone(), \
                self.reward_memory.clone(), \
                self.new_state_memory.clone(), \
                self.terminal_memory.clone()

    def size(self):
        return len(self.buffer)


class ReplayBuffer(object):
    """
    Needs to be modified to account for the number of agents.
    It will receive a batch of states, actions, rewards, next_states, and done
    The batch is the number of agents

    observations = [500, 62]
    
    """
    def __init__(self, max_size, input_shape, chunk_size, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.state_memory = torch.zeros((self.mem_size, chunk_size, n_agents, *input_shape), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, chunk_size, n_agents, *input_shape), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size, chunk_size, n_agents), dtype=torch.float32)
        self.reward_memory = torch.zeros((self.mem_size,chunk_size, n_agents), dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.mem_size, chunk_size, n_agents, dtype=torch.float32)

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