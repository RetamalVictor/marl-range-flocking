import torch
from gym_flock_uw import MultiAgentEnv
from maddpg.agents.ddpg.agent_simple import Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from maddpg.models.DDPG.utils import OUActionNoiseGPU, ReplayBuffer

AGENT_NB = 100
K=4
N_ACTIONS = 2
exp_name = "flock_vxy_4_cohesion"

if __name__ == "__main__":

    env = MultiAgentEnv(agents=AGENT_NB, k=K, collision_distance=0.5, range_start=(0,200), sensor_range=6)
    replay_buffer = ReplayBuffer(max_size=1_000_000, input_shape=[env.k*4],n_agents=AGENT_NB, n_actions=N_ACTIONS)
    noise = OUActionNoiseGPU(mu=torch.zeros(N_ACTIONS).cuda())
    for agent in range(AGENT_NB):
        # Define the prey and predator acceleration networks
        if agent > 3: agent=3
        agent_list = [Agent(
            noise=noise,
            replay_buffer=replay_buffer,
            recurrent=False,
            index=i,
            alpha=0.005,
            beta=0.003,
            # input_dims=[env.k+1],
            input_dims=[env.k*4],
            tau=0.001,
            batch_size=256,
            layer1_size=400,
            layer2_size=300,
            # checkpoint_dir=fr"tmp\{exp_name}\best",
            checkpoint_dir=fr"tmp\{exp_name}\agent_{i}",
            checkpoint_best=fr"tmp\{exp_name}\best") if i < 3 else 
            Agent(
                noise=noise,
                replay_buffer=replay_buffer,
                recurrent=False,
                index=i,
                alpha=0.005,
                beta=0.003,
                input_dims=[env.k*4],
                tau=0.001,
                batch_size=256,
                layer1_size=400,
                layer2_size=300,
                checkpoint_dir=fr"tmp\{exp_name}\agent_{2}",
                # checkpoint_dir=fr"tmp\{exp_name}\best",
                checkpoint_best=fr"tmp\{exp_name}\best")
            for i in range(AGENT_NB)]

        # prey_agent = Agent(agents=1,index=0, alpha=0.001, beta=0.001, input_dims=[env.k+2], tau=0.001, batch_size=256, checkpoint_dir=r"tmp\flock\maddpg")
        # prey_agent.load_models()

        # load models
        for agent_ in agent_list:
            agent_.load_models(False)

        # predator_agent = Agent(alpha=0.001, beta=0.001, input_dims=[6*env.k+2], tau=0.001, batch_size=256,  checkpoint_dir=r"tmp\predator_ddpg")
        # Define the number of episodes and steps per episode
        num_epochs = 2500
        num_test_epochs = 500
        num_episodes = 1
        test=False
        best_score = -np.inf
        action_list = torch.zeros(size=(AGENT_NB,2), dtype=torch.float32).cuda()

        for episode in range(num_episodes):
            observation = env.reset()
            reward_total = 0
            for test_epoch in range(num_test_epochs):
                for index in range(len(agent_list)):
                    action_list[index, :] = agent_list[index].choose_action(observation)

                # Combine the actions and send them to the environment
                # print(action_list)
                next_observation, reward, dones, _ = env.step(action_list, dt=0.05)            
                # _, reward, _, _ = env.step(action_list)
                reward_total += reward.mean().item()
                env.render()
                if dones[1]:
                    print("Collision")
                    break
            print(f"Episode {episode} reward: {reward_total} agent model: {agent}")
            env.close()