import os
import gym, torch
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from copy import copy

"""
Modfiy the reward function
Add GNN
Tweak the loss function to penalize collisions
"""
if not os.path.exists('experiments'):
    os.makedirs('experiments')
exp_name = 'flock'
# writer = SummaryWriter(f'experiments/{exp_name}')

class MultiAgentEnv(gym.Env):
    def __init__(self, agents, k, collision_distance, normalize_distance=False, rigid_boundary=False, range_start=(0,30)):

        """
        :param agents: number of agents
        :param k: number of nearest neighbors
        :param rigid_boundary: if True, agents bounce off the boundary
        :param range_start: range of initial positions
        :param collision_distance: distance at which collision is detected
        :param normalize_distance: if True, normalize the distance between agents
        """

        self.num_particles = agents
        self.k = k
        self.rigid_boundary = rigid_boundary
        self.boundary = range_start[1]
        self.range_start = range_start
        self.collision_distance = collision_distance
        self.normalize_distances = normalize_distance

        self.positions = (range_start[0]-range_start[1])*torch.rand(self.num_particles, 2).cuda() + range_start[1]
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.k + 2,))
        self.memory_size = 4
        self.observation_memory = torch.zeros((self.num_particles, self.memory_size, self.k + 2)).cuda()
        
        
        self.target = torch.tensor([self.boundary//2, self.boundary//2], dtype=torch.float32).cuda()

    def step(self, action, dt = 0.1):
        # update the state
        self._updateState(action, dt)
        self.check_boundary()
        # self.prev_distance_to_near_neighbors = self.distances_to_nearest_neighbors
        self._computeDistances()
        self._computeCollisions()

        obs = self._computeObs()
        dones = self._computeDone()
        rewards = self._computeReward()

        return obs, rewards, dones, {}
    
    def reset(self):
        self.positions = (self.range_start[0]-self.range_start[1])* torch.rand(self.num_particles, 2).cuda() + self.range_start[1]
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
        self.target =(self.range_start[0]-self.range_start[1])* torch.rand(2).cuda() + self.range_start[1]
        self.prev_distance_to_target = torch.norm(self.positions - self.target, dim=1).reshape(-1, 1).cuda()
        # self.prev_distance_to_near_neighbors = torch.zeros(self.num_particles, self.k).cuda()
        self.observation_memory = torch.zeros((self.num_particles, self.memory_size, self.k)).cuda()
        self.check_boundary()
        self._computeDistances()
        self._computeCollisions()
        dones = self._computeDone()
        self.prev_distance_to_near_neighbors = self.distances_to_nearest_neighbors
        if not dones[1]:
            return self._computeObs()
        else:
            return self.reset()

    def _generate_batch(self):        
        batch_input = self.distances_to_nearest_neighbors
        # batch_input = self.distances_to_nearest_neighbors
        # batch_input = torch.cat([ self.distance_to_target, self.target.expand(self.num_particles,-1)] , dim=1)
        # distance_to_bou
        return batch_input

    def _computeObs(self):
        self.observation_memory = torch.roll(self.observation_memory, 1, dims=1)
        self.observation_memory[:, 0, :] = self._generate_batch()
        return copy(self.observation_memory)


    def _computeDistances(self):
        self.positions = self.positions.detach()
        if self.normalize_distances:
            magnitudes = torch.norm(self.positions, dim=1)
            normalized_positions = self.positions / torch.max(magnitudes)
            # Distance calculation
            self.distances = torch.norm(normalized_positions[:,None ] - normalized_positions[:], dim=2)
        else:
            # Distance calculation
            self.distances = torch.norm(self.positions[:,None ] - self.positions[:], dim=2)
        
        # Nearest neighbors
        distances_to_nearest_neighbors, nearest_neighbors = torch.topk(-self.distances, self.k + 1, dim=1)
        self.distances_to_nearest_neighbors = -distances_to_nearest_neighbors[:, 1:]
    
    def distance_regions(self, distance_tensor):
            """
            Squared error and extra punishment for being to close
            Let distance_array be a vector of distances. 
            The function defines a penalty y based on the distance in distance_array. 
            The penalty is calculated as follows:

            if distance_array[i] < 0, then y[i] = (distance_array[i] + 0.1)^2

            if distance_array[i] >= 0, then y[i] = distance_array[i]^2

            The logic in the function is to add a squared error penalty for the distances in distance_array, 
            with an extra punishment for being too close (i.e. for distances less than 0). 
            The extra punishment is applied by adding 0.1 to the distance before squaring it.

            Parameters
            ----------
            distance_tensor : torch.tensor shape(agents, k)

            Returns
            -------
            torch.tensor shape(agents, 1)
                The reward value for each agent.
            """

            y = torch.zeros(distance_tensor.shape).cuda()
            y += (distance_tensor < 0) * (torch.pow(distance_tensor - 0.1, 2))
            y += (distance_tensor >= 0) * (torch.pow(distance_tensor, 2))
            return y

    def _computeCollisions(self):
        self.collisions = torch.where(
            self.distances_to_nearest_neighbors < self.collision_distance, 1, 0
            )

    def _computeCollisionPenalty(self):
        # collisions
        collisions = torch.any(self.collisions, 1)
        return torch.where(collisions==True, -5, 0.01)

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        np.array[float] shape(agents, 1)
            The reward value for each agent.
        """
        # distance_rewards = self.distance_regions(self.distances_to_nearest_neighbors)
        collision_penalty = self._computeCollisionPenalty()
        # distance_to_target = torch.norm(self.positions - self.target, dim=1).reshape(-1, 1)
        # reach_target = torch.where(distance_to_target < self.prev_distance_to_target, 1, 0)
        
        # center_of_mass = torch.mean(self.positions, dim=0)
        # distance_to_center_of_mass = torch.norm(self.positions - center_of_mass, dim=1).reshape(-1, 1)
        # cohesion_reward = 

        rewards = collision_penalty.reshape(-1,1)
        return rewards
  
    
    def check_boundary(self):

        if self.rigid_boundary:
            self.positions[:, 0] = torch.where(self.positions[:, 0] < self.boundary, self.positions[:, 0], self.boundary)
            self.positions[:, 0] = torch.where(self.positions[:, 0] > 0, self.positions[:, 0], 0)
            self.positions[:, 1] = torch.where(self.positions[:, 1] < self.boundary, self.positions[:, 1], self.boundary)
            self.positions[:, 1] = torch.where(self.positions[:, 1] > 0, self.positions[:, 1], 0)
        
        else:
            self.positions[:, 0] = torch.where(self.positions[:, 0] < self.boundary, self.positions[:, 0], 0.001)
            self.positions[:, 0] = torch.where(self.positions[:, 0] > 0, self.positions[:, 0], self.boundary)

            self.positions[:, 1] = torch.where(self.positions[:, 1] < self.boundary, self.positions[:, 1], 0.001)
            self.positions[:, 1] = torch.where(self.positions[:, 1] > 0, self.positions[:, 1], self.boundary)     

    def _computeDone(self):
        """
        Computes the current done value(s).
        returns a tuple of (dones, all_done)
        dones is a np.array of booleans of shape (agents, 1)
        all_done is a boolean
        """
        # Check if any collision
        dones = torch.any(self.collisions, 1)
        return (dones, torch.any(self.collisions).item())

    def _updateState(self, action, dt):
        # update the position and velocity based on the action
        self.velocities += action * dt
        # Multiply accelerations of the predators by 2
        self.velocities = self.velocities/torch.norm(self.velocities, dim=1, keepdim=True)
        # self.velocities *= 2
        self.positions += (self.velocities * dt)


    def render(self):
        plt.ion()
        diff = self.positions - self.velocities

        ind = torch.zeros(self.num_particles)

        plt.clf()
        square = plt.Rectangle((0, 0), self.boundary, self.boundary, fill=False)
        plt.gca().add_patch(square)
        plt.quiver(diff[:, 0].cpu().detach(), diff[:, 1].cpu().detach(), self.velocities[:, 0].cpu().detach(), 
            self.velocities[:, 1].cpu().detach(), ind.float().cpu().detach().numpy(), cmap ='coolwarm')
        plt.scatter(self.target[0].cpu().detach(), self.target[1].cpu().detach(), color='red', s=100)

        plt.draw()
        plt.axis([0-10, self.boundary+10, 0-10, self.boundary+10])
        plt.pause(0.001)

    def close(self):
        plt.close()


def test_env():
    nb_agents = 2
    k=1

    # Initialize the environment
    env = MultiAgentEnv(nb_agents,k,collision_distance=1, normalize_distance=False, range_start=(0,15) )

    # Define the number of episodes and steps per episode
    num_episodes = 1
    num_epochs = 100
    mode = 'train'

    for episode in range(num_episodes):
        observation = env.reset()
        actions = np.array([[1,0],[-1,0]])
        for epoch in range(num_epochs):
            with torch.no_grad():
                # Get the actions for the prey and predator
                prey_action = torch.from_numpy(np.random.uniform(-1, 1, (nb_agents,2))).float().cuda()
                # prey_action = torch.from_numpy(actions).float().cuda()
                obs, reward, dones, _ = env.step(prey_action)
                print(obs[0])
            # if dones[1]:
            #     observation = env.reset()
            #     print("reset")
            env.render()
        env.close()

if __name__ == "__main__":
    test_env()