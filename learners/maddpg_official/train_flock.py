import time, os
import numpy as np
import argparse

from tqdm import tqdm
from gym_flock_v2 import make_env
from MADDPG import SuperAgent

from torch.utils.tensorboard import SummaryWriter


def main(args):

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp_name))
    env = make_env(args)
    super_agent = SuperAgent(args, env)
    MAX_STEPS = args.max_steps

    scores = []

    # Checkpoint loading --- FOR LATER

    for n_game in range(args.max_games):
        start_time = time.time()
        obs = env.reset()
        finish = False
        score = 0
        step = 0
        while not finish and step <  MAX_STEPS: # Done comes as tuple (Tensor, bool) from the environment
            actors_state = obs["actors"]
            actors_action = super_agent.get_actions(actors_state)
            next_obs, reward, done, _ = env.step(actors_action)
            
            state = obs["critic"]
            next_state = next_obs["critic"]
            
            super_agent.replay_buffer.add_record(obs["actors"], next_obs['actors'], actors_action, state, next_state, reward, done[0])
            finish = done[1]
            obs  = next_obs
            score += sum(reward)/ env.num_particles
            step += 1
            if not super_agent.replay_buffer.check_buffer_size() and step % 100 == 0:
                print(f"Buffer at {round(super_agent.replay_buffer.buffer_counter / super_agent.replay_buffer.min_size_buffer * 100, 2)}, [{super_agent.replay_buffer.buffer_counter}, {super_agent.replay_buffer.min_size_buffer}]")
                finish = False
                obs = env.reset()
                # step = 0


        if super_agent.replay_buffer.check_buffer_size():
            super_agent.train()
        writer.add_scalar("Train/Score", score.item(), n_game)
        writer.add_scalar("Train/Average score", np.mean(scores[-10:]), n_game)
        writer.add_scalar("Train/Time taken", time.time() - start_time, n_game)
        writer.add_scalar("Train/Buffer size", super_agent.replay_buffer.buffer_counter, n_game)
        super_agent.replay_buffer.update_n_games()
        scores.append(score.item())
        

        # wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter, 
        #            "Average reward": round(np.mean(scores[-10:]), 2), \
        #                   "Time taken": round(time.time() - start_time, 2), 'Max steps': MAX_STEPS})
        
        if (n_game + 1) % args.evaluation_freq == 0 and super_agent.replay_buffer.check_buffer_size():
            obs = env.reset()
            done = [False for index in range(super_agent.n_agents)]
            score = 0
            step = 0
            while not done[1] and step <  MAX_STEPS: # Done comes as tuple (Tensor, bool) from the environment
                actors_state = obs["actors"]
                actors_action = super_agent.get_actions(actors_state)
                next_obs, reward, done, _ = env.step(actors_action)
                obs  = next_obs
                score += sum(reward)
                step += 1
            writer.add_scalar("Evaluation/Score", score, n_game)
            writer.add_scalar("Evaluation/Average score", np.mean(scores[-10:]), n_game)

        if (n_game + 1) % args.save_freq == 0 and super_agent.replay_buffer.check_buffer_size():
            print("saving weights and replay buffer...")
            super_agent.save()
            print("saved")


if __name__ == "__main__":
    date_now = time.strftime("%Y%m%d%H%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_agents", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--collision_distance", type=float, default=3)
    parser.add_argument("--normalize_distance", type=bool, default=False)
    parser.add_argument("--range_start", type=tuple, default=(0, 50))
    parser.add_argument("--max_games", type=int, default=1_000_000)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--evaluation_freq", type=int, default=100)
    parser.add_argument("--ou_theta", type=float, default=0.15)
    parser.add_argument("--ou_sigma", type=float, default=0.2)
    parser.add_argument("--ou_mu", type=float, default=0)
    parser.add_argument("--ou_dt", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    # add writer params
    parser.add_argument("--log_dir", type=str, default="maddpgv2/logs/")
    parser.add_argument("--exp_name", type=str, default="training_1")
    parser.add_argument("--save_dir", type=str, default=f"maddpgv2/checkpoints/{date_now}")

    
    args = parser.parse_args()
    main(args)