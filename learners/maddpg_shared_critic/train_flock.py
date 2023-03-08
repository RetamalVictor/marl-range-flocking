import os, time, torch, yaml
import gym 
from tqdm import tqdm
import numpy as np
# from gym_flock import MultiAgentEnv
from gym_flock_uw import MultiAgentEnv
from torch.utils.tensorboard import SummaryWriter

# from maddpg.agents.ddpg.agent_simple import Agent
from maddpg.agents.ddpg.agent_simple_shared_critic import Agent


def main():

    exp_name = "flock_uw_5"
    AGENT_NB = 8
    NEAREST_NEIGHBORS = 3
    COLLISION_DISTANCE = 3
    RECURRENT = False
    N_ACTIONS = 2
    ALPHA = 3e-4
    BETA = 3e-4
    TAU = 0.001
    LAYER1_SIZE = 400
    LAYER2_SIZE = 300
    BATCH_SIZE = 256
    CHECKPOINT_DIR = fr"tmp\{exp_name}"

    if RECURRENT:
        from maddpg.models.ddpg_RNN.utils import OUActionNoiseGPU, ReplayBuffer
    else:
        from maddpg.models.DDPG.utils import OUActionNoiseGPU, ReplayBuffer
        from maddpg.models.DDPG.DDPG_network import CriticNetwork

     # Initialize the environment
    env = MultiAgentEnv(agents=AGENT_NB, k=NEAREST_NEIGHBORS, collision_distance=COLLISION_DISTANCE, range_start=(0,50),sensor_range=7)
    INPUT_DIMS = [env.k*4]
    # Define the agents
    replay_buffer = ReplayBuffer(max_size=1_000_000, input_shape=[env.k*4],n_agents=AGENT_NB, n_actions=N_ACTIONS)
    noise = OUActionNoiseGPU(mu=torch.zeros(N_ACTIONS).cuda())
    shared_critic = CriticNetwork(
            BETA,
            INPUT_DIMS,
            LAYER1_SIZE,
            LAYER2_SIZE,
            n_actions=N_ACTIONS,
            name="Critic",
            chkpt_dir=os.path.join(CHECKPOINT_DIR, "critic"),
            chkpt_best_dir=os.path.join(CHECKPOINT_DIR, "best")
        )

    agent_list = [Agent(
        shared_critic=shared_critic,
        replay_buffer=replay_buffer,
        noise=noise,
        recurrent=RECURRENT,
        index=i,
        alpha=ALPHA,
        beta=BETA, 
        input_dims=INPUT_DIMS, 
        layer1_size=LAYER1_SIZE,
        layer2_size=LAYER2_SIZE,
        tau=TAU, 
        batch_size=256, 
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, "agent_" + str(i)),
        checkpoint_best=os.path.join(CHECKPOINT_DIR, "best"))
        for i in range(AGENT_NB)]

    writer = SummaryWriter(log_dir=fr"tmp\runs\{exp_name}")
    with open(F"tmp\{exp_name}\exp_info.txt", "w") as f:
        f.write("Experiment: " + exp_name + "\nRecurrent: "+ str(RECURRENT) +"\nAgents: " + str(AGENT_NB) + "\nNearest Neighbors: " + str(NEAREST_NEIGHBORS) + "\nCollision Distance: " + str(COLLISION_DISTANCE) + "\n")
        f.write("Adding sensing range.\n")
        f.write("Vx_Vy.\n")
        f.write("Adding cohesion.\n")
        f.write("Layers to 400 and 300.\n")


    # Define the number of episodes and steps per episode
    num_epochs = 150
    num_test_epochs = 100
    num_episodes = 5000
    test=False
    best_score = -np.inf
    action_list = torch.zeros(size=(AGENT_NB,2), dtype=torch.float32).cuda()
    last_time = time.time()

    # Fill replay buffer with random actions
    print("Filling replay buffer")
    observation = env.reset()
    for episode in tqdm(range((1_000_000//AGENT_NB) + AGENT_NB)):
    # while replay_buffer.mem_cntr < replay_buffer.mem_size:
        action_list = torch.rand(size=(AGENT_NB,2), dtype=torch.float32).cuda()
        next_observation, reward, dones, _ = env.step(action_list)
        replay_buffer.store_transitions(observation.reshape(AGENT_NB,-1), action_list, reward, next_observation.reshape(AGENT_NB,-1), dones[0].long())
        observation = next_observation
        if dones[1]:
            observation = env.reset()
        # if episode % 1000 == 0:
        #     print("Filling replay buffer: %i" % replay_buffer.mem_cntr)
        if replay_buffer.mem_cntr > replay_buffer.mem_size:
            break

    print("Replay buffer filled with %i transitions" % replay_buffer.mem_cntr)
    print("Starting training -----------")
    for episode in range(num_episodes):
        reward_total = 0
        reward_per_agent = torch.zeros((AGENT_NB,1)).cuda()
        ac_loss_total = 0
        ct_loss_total = 0
        # observation = env.generate_batch() # Generating observation for the first time
        observation = env.reset()
        for epoch in range(1, num_epochs):
            # Get the actions for the prey and predator
            for index in range(len(agent_list)):
                action_list[index, :] = agent_list[index].choose_action(observation.reshape(AGENT_NB,-1))

            next_observation, reward, dones, _ = env.step(action_list)
            # Alternate between training the prey and predator networks
            replay_buffer.store_transitions(observation.reshape(AGENT_NB,-1), action_list, reward, next_observation.reshape(AGENT_NB,-1), dones[0].long())
            for agent in range(len(agent_list)):
                ac_loss, ct_loss, ready = agent_list[agent].learn()
                ac_loss_total += ac_loss
                ct_loss_total += ct_loss

            reward_total += (reward.mean().item())
            reward_per_agent += reward
            observation = next_observation

            if dones[1]:
                observation = env.reset()
                print("Collision")
                break

        print("Episode: %i Reward : %f" % (episode, reward_total/epoch))
        writer.add_scalar('Rewards/Reward_episode', reward_total/ epoch, episode)
        writer.add_scalar('Loss/Actor', ac_loss_total   / epoch, episode)
        writer.add_scalar('Loss/Critic', ct_loss_total  / epoch, episode)

        if last_time + 300 < time.time():
            last_time = time.time()
            print("Saving models, On time")
            for index in range(len(agent_list)):
                agent_list[index].save_models()

        # Get best agent reward
        best_agent_reward = reward_per_agent.max().item()
        writer.add_scalar('Rewards/Best_agent', best_agent_reward/ epoch, episode)

        if (best_agent_reward/ epoch) > best_score:
            best_score = (best_agent_reward/ epoch)
            agent_list[torch.argmax(reward_per_agent)].save_models_best()
            print("Best score: %f" % best_score)

        env.reset()
        if test and episode % 10 == 0:
            observation = env.reset()
            for test_epoch in range(num_test_epochs):

                for index in range(len(agent_list)):
                    action_list[index, :] = agent_list[index].choose_action(observation)                

                observation, reward, _, _ = env.step(action_list)
                env.render()
            env.close()

if __name__ == "__main__":
   main()