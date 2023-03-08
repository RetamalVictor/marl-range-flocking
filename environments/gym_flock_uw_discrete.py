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
if not os.path.exists("experiments"):
    os.makedirs("experiments")
exp_name = "flock_vxy_1"
# writer = SummaryWriter(f'experiments/{exp_name}')


class MultiAgentEnv(gym.Env):
    def __init__(
        self,
        agents,
        k=4,
        collision_distance=3,
        normalize_distance=False,
        rigid_boundary=False,
        range_start=(0, 100),
        sensor_range=7,
        max_linear_velocity=2.5,
        desired_distance = 15
    ):

        """
        :param agents: number of agents
        :param k: number of nearest neighbors
        :param rigid_boundary: if True, agents bounce off the boundary
        :param range_start: range of initial positions
        :param collision_distance: distance at which collision is detected
        :param normalize_distance: if True, normalize the distance between agents
        """
        super(MultiAgentEnv, self).__init__()
        self.num_particles = agents
        self.k = k
        self.rigid_boundary = rigid_boundary
        self.boundary = range_start[1]
        self.desired_distance = desired_distance
        self.range_start = range_start
        self.sensor_range = sensor_range
        self.max_linear_velocity = max_linear_velocity
        self.collision_distance = collision_distance
        self.collision_temp = collision_distance
        self.normalize_distances = normalize_distance

        # Dictionary of actions containing the means for the sampled actions
        # The means are for linear and angular speed.
        # In total there is 15 different actions
        # Linear is [0.4, 0.9, 1.4]
        # Angular is [-1.4, -0.7, 0, 0.7, 1.4 ]
        self.action_dictionary = {
            0:  [0.2, -1.2],
            1:  [0.2, -0.5],
            2:  [0.2, 0],
            3:  [0.2, 0.5],
            4:  [0.2, 1.2],
            5:  [0.6, -1.2],
            6:  [0.6, -0.5],
            7:  [0.6, 0],
            8:  [0.6, 0.5],
            9:  [0.6, 1.2],
            # 10: [1.2, -1.2],
            # 11: [1.2, -0.5],
            # 12: [1.2, 0],
            # 13: [1.2, 0.5],
            # 14: [1.2, 1.2],
        }
        # self.action_dictionary_v_1 = {
        #     0:  [0.2, -1.2],
        #     1:  [0.2, -0.5],
        #     2:  [0.2, 0],
        #     3:  [0.2, 0.5],
        #     4:  [0.2, 1.2],
        #     5:  [0.6, -1.2],
        #     6:  [0.6, -0.5],
        #     7:  [0.6, 0],
        #     8:  [0.6, 0.5],
        #     9:  [0.6, 1.2],
        #     10: [1.2, -1.2],
        #     11: [1.2, -0.5],
        #     12: [1.2, 0],
        #     13: [1.2, 0.5],
        #     14: [1.2, 1.2],
        # }

        self.positions = (range_start[0] - range_start[1]) * torch.rand(
            self.num_particles, 2
        ).cuda() + range_start[1]
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
        self.action_space = list(spaces.Discrete(self.k) for _ in range(self.num_particles))
        self.observation_space = list(spaces.Box(low=0, high=self.sensor_range, shape=(self.k,)) for _ in range(self.num_particles))
        # self.memory_size = 4
        # self.observation_memory = torch.zeros(
        #     (self.num_particles, self.memory_size, self.k + 2)
        # ).cuda()

        self.headings = torch.rand(self.num_particles, 2).cuda()
        # self.target = torch.tensor(
        #     [self.boundary // 2, self.boundary // 2], dtype=torch.float32
        # ).cuda()

    def step(self, action, dt=0.1):
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
        self.positions = (self.range_start[0] - self.range_start[1]) * torch.rand(
            self.num_particles, 2
        ).cuda() + self.range_start[1]
        # self.positions = (self.range_start[0] - self.range_start[1]//2) * torch.rand(
        #     self.num_particles, 2
        # ).cuda() + self.range_start[1]//2
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
        self.prev_headings = torch.zeros(self.num_particles).cuda()
        self.headings =  (0 - np.pi/1.2) * torch.rand(self.num_particles).cuda() + np.pi/1.2
        # self.headings =  torch.rand(self.num_particles).cuda()
        self.target = (self.range_start[0] - self.range_start[1]) * torch.rand(
            2
        ).cuda() + self.range_start[1]
        self.prev_distance_to_target = (
            torch.norm(self.positions - self.target, dim=1).reshape(-1, 1).cuda()
        )
        # self.prev_distance_to_near_neighbors = torch.zeros(self.num_particles, self.k).cuda()
        # self.observation_memory = torch.zeros(
        #     (self.num_particles, self.memory_size, self.k)
        # ).cuda()
        self.collision_distance = 4
        self.check_boundary()
        self._computeDistances()
        self._computeCollisions()
        dones = self._computeDone()
        self.prev_distance_to_near_neighbors = self.distances_to_nearest_neighbors
        if not dones[1]:
            self.collision_distance = self.collision_temp
            return self._computeObs()

        else:
            return self.reset()

    def _generate_batch(self):
        batch_input = self.distances_to_nearest_neighbors
        # batch_input = self.distances_to_nearest_neighbors
        # batch_input = torch.cat([ self.distance_to_target, self.target.expand(self.num_particles,-1)] , dim=1)
        # distance_to_bou
        return batch_input

    # def _computeObs(self):
    #     self.observation_memory = torch.roll(self.observation_memory, 1, dims=1)
    #     self.observation_memory[:, 0, :] = self._generate_batch()
    #     return copy(self.observation_memory)
    
    def _computeObs(self):
        return self._generate_batch()

    def _computeDistances(self):
        self.positions = self.positions.detach()
        if self.normalize_distances:
            magnitudes = torch.norm(self.positions, dim=1)
            normalized_positions = self.positions / torch.max(magnitudes)
            # Distance calculation
            self.distances = torch.norm(
                normalized_positions[:, None] - normalized_positions[:], dim=2
            )
        else:
            # Distance calculation
            self.distances = torch.norm(
                self.positions[:, None] - self.positions[:], dim=2
            )

        # Nearest neighbors
        distances_to_nearest_neighbors, nearest_neighbors = torch.topk(
            -self.distances, self.k + 1, dim=1
        )
        self.distances_to_nearest_neighbors = torch.clamp(-distances_to_nearest_neighbors[:, 1:], min=0, max=self.sensor_range)

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

    def _computeDesiredDistanceReward(self):
        desired_distance_error = torch.abs(self.distances_to_nearest_neighbors - self.desired_distance)
        normalized_desired_distance_error = desired_distance_error / torch.norm(desired_distance_error)
        return (1 - normalized_desired_distance_error)/100

    def _computeCollisions(self):
        self.collisions = torch.where(
            self.distances_to_nearest_neighbors < self.collision_distance, 1, 0
        )

    def _computeCollisionPenalty(self):
        # collisions
        collisions = torch.any(self.collisions, 1)
        return torch.where(collisions == True, -9, 0)

    def _computeCenterOfMassReward(self):
        # center of mass
        self.center_of_mass = torch.mean(self.positions, dim=0)
        distance_to_center_of_mass = torch.norm(
            self.positions - self.center_of_mass, dim=1
        ).reshape(-1, 1)
        return torch.where(distance_to_center_of_mass < self.collision_distance*4, 0.1, 0)
        # return torch.where(distance_to_center_of_mass < self.collision_distance*4, (distance_to_center_of_mass**2)/550, 0)

    
    def _computePenaltyAngularVelocity(self):
        diff = torch.abs(self.prev_headings - self.headings)
        self.prev_headings = torch.clone(self.headings)
        return torch.where(diff > 0.27, -0.01, 0.001)
    
    def _computeAlignmentReward(self):
        # alignment
        alignment = torch.sum(self.headings)/self.num_particles
        alignment_error = torch.abs(alignment - self.headings)
        return torch.where(alignment_error > 0.20, 0, 0.1)

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        np.array[float] shape(agents, 1)
            The reward value for each agent.
        """
        collision_penalty = self._computeCollisionPenalty()
        center_of_mass_reward = self._computeCenterOfMassReward()
        angular_velocity_penalty = self._computePenaltyAngularVelocity()
        alignment_reward = self._computeAlignmentReward()
        # distance_regions = self.distance_regions(self.distances_to_nearest_neighbors)
        # desired_distance_reward = torch.sum(self._computeDesiredDistanceReward(), axis=1)

        rewards = collision_penalty.reshape(-1, 1) + alignment_reward.reshape(-1,1) #+ center_of_mass_reward.reshape( -1, 1) + angular_velocity_penalty.reshape(-1, 1) #+ desired_distance_reward.reshape(-1,1)#+ distance_regions.reshape(-1, 1) 
        return rewards

    def check_boundary(self):

        if self.rigid_boundary:
            self.positions[:, 0] = torch.where(
                self.positions[:, 0] < self.boundary,
                self.positions[:, 0],
                self.boundary,
            )
            self.positions[:, 0] = torch.where(
                self.positions[:, 0] > 0, self.positions[:, 0], 0
            )
            self.positions[:, 1] = torch.where(
                self.positions[:, 1] < self.boundary,
                self.positions[:, 1],
                self.boundary,
            )
            self.positions[:, 1] = torch.where(
                self.positions[:, 1] > 0, self.positions[:, 1], 0
            )

        else:
            self.positions[:, 0] = torch.where(
                self.positions[:, 0] < self.boundary, self.positions[:, 0], 0.001
            )
            self.positions[:, 0] = torch.where(
                self.positions[:, 0] > 0, self.positions[:, 0], self.boundary
            )

            self.positions[:, 1] = torch.where(
                self.positions[:, 1] < self.boundary, self.positions[:, 1], 0.001
            )
            self.positions[:, 1] = torch.where(
                self.positions[:, 1] > 0, self.positions[:, 1], self.boundary
            )

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

    def _updateState(self, action, dt, heading:bool = True):
        """
        Actions is a tensor of shape (agents, 1)
        """
        # Translate from action dictionary to linear and angular means
        means_linear_velocity = torch.tensor([self.action_dictionary[int(act)] for act in action])[:,0].cuda()
        means_angular_velocity = torch.tensor([self.action_dictionary[int(act)] for act in action])[:,1].cuda()

        # sample the linear and angular for every agent from normal distribution
        linear_velocity = torch.normal(mean=means_linear_velocity, std=0.1)
        angular_velocity = torch.normal(mean=means_angular_velocity, std=0.1)



        if heading:
            # linear_velocity = action[:, 0]
            # angular_velocity = action[:, 1]

            # angular_velocity = torch.clamp(angular_velocity, -np.pi / 2, np.pi / 2)
            angular_velocity = torch.clamp(angular_velocity, -0.025, 0.025)
            # self.prev_angular_velocity = angular_velocity
            self.headings += angular_velocity * dt
            # clip linear velocity to be greater than 0
            linear_velocity = torch.clamp(linear_velocity, 5e-6, self.max_linear_velocity)
            # clip angular velocity to be between -pi and pi using torch.clamp

            # convert linear and angular velocity to x and y velocity
            vx = linear_velocity * torch.cos(self.headings)
            vy = linear_velocity * torch.sin(self.headings)
            self.velocities = torch.stack((vx, vy), dim=1).cuda()

        if not heading:
            self.velocities = action.cuda()
        # normalize the velocity
        self.velocities = self.velocities / torch.norm(
            self.velocities, dim=1, keepdim=True
        )

        self.velocities = torch.nan_to_num(self.velocities)

        # update the position and velocity based on the action
        self.velocities *= dt
        self.positions += self.velocities

    def render(self):
        plt.ion()
        diff = self.positions - self.velocities

        ind = torch.zeros(self.num_particles)

        plt.clf()
        square = plt.Rectangle((0, 0), self.boundary, self.boundary, fill=False)
        plt.gca().add_patch(square)
        plt.quiver(
            diff[:, 0].cpu().detach(),
            diff[:, 1].cpu().detach(),
            self.velocities[:, 0].cpu().detach(),
            self.velocities[:, 1].cpu().detach(),
            ind.float().cpu().detach().numpy(),
            cmap="coolwarm",
            linewidth=2,
            # scale=2,
        )
        # plot positions
        plt.scatter(
            self.positions[:, 0].cpu().detach(),
            self.positions[:, 1].cpu().detach(),
            color="black",
            s=20,
        )
        # plot sensing range for one agent
        circle = plt.Circle(
            (self.positions[0, 0].cpu().detach(), self.positions[0, 1].cpu().detach()),
            self.sensor_range,
            fill=False,
        )
        plt.gca().add_patch(circle)

        # plot center of mass
        # plt.scatter(
        #     self.center_of_mass[0].cpu().detach(),
        #     self.center_of_mass[1].cpu().detach(),
        #     color="red",
        #     s=100,
        # )
        # plot a circle around the center of mass
        # circle = plt.Circle(
        #     (self.center_of_mass[0].cpu().detach(), self.center_of_mass[1].cpu().detach()),
        #     self.collision_distance*4,
        #     fill=False,
        # )
        # plt.gca().add_patch(circle)



        # plt.scatter(
        #     self.target[0].cpu().detach(),
        #     self.target[1].cpu().detach(),
        #     color="red",
        #     s=100,
        # )

        plt.draw()
        plt.axis([0 - 10, self.boundary + 10, 0 - 10, self.boundary + 10])
        plt.pause(0.001)
        # plt.pause(0.5)

    def close(self):
        plt.close()


def test_env():
    nb_agents = 8
    k = 3

    # Initialize the environment
    env = MultiAgentEnv(
        nb_agents,
        k,
        collision_distance=1,
        normalize_distance=False,
        range_start=(0, 50),
    )

    # Define the number of episodes and steps per episode
    num_episodes = 1
    num_epochs = 150
    mode = "train"

    for episode in range(num_episodes):
        observation = env.reset()
        actions = np.array([[1, 0], [-1, 0]])
        for epoch in range(num_epochs):
            with torch.no_grad():
                # Get the actions for the prey and predator
                prey_action = (
                    torch.from_numpy(
                        np.array(
                            [
                                # np.zeros((nb_agents)),
                                # np.zeros((nb_agents)),
                                np.random.randint(0, 14, nb_agents),
                                # np.random.uniform(-2, 2, nb_agents),
                            ]
                        ).reshape(nb_agents, 1)
                    )
                    .long()
                    .cuda()
                )
                # print(env.headings)
                # prey_action = torch.from_numpy(actions).float().cuda()
                # print(prey_action)
                obs, reward, dones, _ = env.step(prey_action)
                print(reward)
            if dones[1]:
                observation = env.reset()
                print("reset")
            env.render()
        env.close()


if __name__ == "__main__":
    test_env()
