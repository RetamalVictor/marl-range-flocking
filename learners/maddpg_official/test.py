"""
This file contain unit test for the environment, agent, maddpg, replay buffer and model
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import os, time

from pprint import pprint
from gym_flock_v2 import make_env
from MADDPG import SuperAgent
from memory import ReplayBufferMaddpg
from utils import OrnsteinUhlenbeckProcess
from agent import Agent


class TestAgent(unittest.TestCase):
    def test_agent(self):
        """
        Test the initialization of the agent.
        Test the action shape
        test the actor and critic model. Check shape of inputs and ouptuts
        """
        args = {
            "nb_agents": 4,
            "k": 2,
            "collision_distance": 1,
            "normalize_distance": False,
            "range_start": (0, 50),
        }
        env = make_env(args)
        obs = env.reset()
        agent = Agent(env, 0, OrnsteinUhlenbeckProcess(size=2, theta=0.15, mu=0, sigma=0.2))
        self.assertEqual(agent.actor.input_dim, 4)
        self.assertEqual(agent.actor.nb_actions, 2)
        self.assertEqual(agent.critic.input_dim, 8)
        self.assertEqual(agent.critic.nb_actions, 2)

        action = agent.choose_action(obs["actors"])
        self.assertEqual(action.shape, (2,))




class TestEnv(unittest.TestCase):

    def test_env(self):
        """
        Test whether the environment is working
        test the obs shape. It should be a dict with keys "critic" and "actor"
        actor: (nb_agents, k)
        critic: (nb_agents, k + 4)
        """
        args = {
            "nb_agents": 4,
            "k": 2,
            "collision_distance": 1,
            "normalize_distance": False,
            "range_start": (0, 50),
        }
        env = make_env(args)
        _ = env.reset()
        action = (
            torch.from_numpy(
                np.array(
                    [
                        np.random.uniform(-3.2, 3.2, args.nb_agents),
                        np.random.uniform(-2, 2, args.nb_agents),
                    ]
                ).reshape(args.nb_agents, 2)
            )
            .float()
            .cuda()
        )
        obs, reward, dones, _ = env.step(action)
        self.assertEqual(obs["critic"].shape, (args.nb_agents, args.k + 4))
        self.assertEqual(obs["actors"].shape, (args.nb_agents, args.k))
        env.close()
