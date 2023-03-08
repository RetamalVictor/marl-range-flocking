"""
Load the models from the folder "tmp\ddpg" and run them in the environment.
"""
import sys
sys.path.append(r"C:\Users\victo\Desktop\VU master\Multiagent\multi-rl-crowd-sim-mains")
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time, os
import numpy as np
import argparse

from tqdm import tqdm
from gym_flock_v2 import make_env
from MADDPG import SuperAgent

from torch.utils.tensorboard import SummaryWriter

def main(args):
    
        # writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp_name))
        env = make_env(args)
        super_agent = SuperAgent(args, env)
        MAX_STEPS = args.max_steps
    
        scores = []

        # Checkpoint loading 
        # super_agent.load()
        for i in range(9):
            # super_agent.load_single_checkpoint(fr"C:\Users\victo\Desktop\VU master\Multiagent\multi-rl-crowd-sim-main\maddpgv2_rnn\checkpoints\maddpg_rnn_collv3_2023-02-23-2115\agent_number_{i}_actor_ddpg.pt")
            super_agent.load_scaled_checkpoint(r"C:\Users\victo\Desktop\VU master\Multiagent\multi-rl-crowd-sim-main\maddpgv2_rnn\checkpoints\maddpg_rnn_casv2_2023-02-24-0731 copy", total=9)
        
            for n_game in range(args.max_games):
                start_time = time.time()
                obs = env.reset()
                actors_state = obs["actors"]
                finish = False
                score = 0
                step = 0
                obs = env.reset()
                done = [False for index in range(super_agent.n_agents)]
                score = 0
                step = 0
                # actors_state = obs["actors"]
                hidden = super_agent.init_hidden()
                super_agent.reset_random_process()
                while not done[1] and step <  MAX_STEPS: # Done comes as tuple (Tensor, bool) from the environment
                    actors_state = obs["actors"]
                    actors_action, hidden = super_agent.get_actions(actors_state, hidden, test=True)
                    next_obs, reward, done, _ = env.step(actors_action, dt=0.1)
                    obs  = next_obs
                    score += sum(reward)
                    step += 1
                    env.render()
                    if done[1]:
                        print("Collision")
                print(f"Game {n_game} finished after {step} steps with a score of {score.item()}")
                # wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter, 
                #            "Average reward": round(np.mean(scores[-10:]), 2), \
                #                   "Time taken": round(time.time() - start_time, 2), 'Max steps': MAX_STEPS})
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_agents", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--collision_distance", type=float, default=0.5)
    parser.add_argument("--normalize_distance", type=bool, default=False)
    parser.add_argument("--range_start", type=tuple, default=(0, 30))
    parser.add_argument("--sensor_range", type=float, default=14)
    parser.add_argument("--max_games", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--save_freq", type=int, default=250)
    parser.add_argument("--evaluation_freq", type=int, default=25)
    parser.add_argument("--ou_theta", type=float, default=0.15)
    parser.add_argument("--ou_sigma", type=float, default=0.01)
    parser.add_argument("--ou_sigma_min", type=float, default=0.001)
    parser.add_argument("--ou_mu", type=float, default=0)
    parser.add_argument("--ou_dt", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--min_size_buffer", type=int, default=8_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    # add writer params
    parser.add_argument("--log_dir", type=str, default="maddpgv2/logs/")
    parser.add_argument("--exp_name", type=str, default="test_5")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\victo\Desktop\VU master\Multiagent\multi-rl-crowd-sim-main\maddpgv2")

    
    args = parser.parse_args()
    main(args)