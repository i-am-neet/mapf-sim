#!/usr/bin/env python3
from mapf_env_node import StageEnv
import os
import argparse
import torch
import random
import numpy as np
import utils
import time
import itertools
import datetime
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

arg_fmt = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description='Reinforcement Learning Arguments',
                                 formatter_class=arg_fmt)
parser.add_argument('-crn', '--current-robot-num', type=int,
                    help='The number of current robot. (For env)'
                    )
parser.add_argument('-rn', '--robots-num', type=int,
                    help='The amount of all robots. (For env)'
                    )
parser.add_argument('--robot-diameter', default=0.25, type=float,
                    help='The diameter of robot (default is according to TB3) (For env)'
                    )
parser.add_argument('--map-resolution', default=0.01, type=float,
                    help='The resolution of map (For env)'
                    )
parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--goals_init_file', type=str, default=os.environ.get('HOME')+"/mapf_ws/src/gazebo/robot_gazebo/config/init_goals.cfg",
                    help="config file of goals' init info (default: '$HOME/mapf_ws/src/gazebo/robot_gazebo/config/init_goals.cfg')")
parser.add_argument('--resize_observation', type=int, default=128, metavar='N',
                    help='Resize observation size (default: 128)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
args = parser.parse_args()

## Tmp
# goals = [(2.1, 2.1, 0.0), (-1.8, 1.8, 0.0), (1.5, -1.5, 0.0), (-2.1, -2.1, 0.0)]
goals = []

with open(args.goals_init_file) as file:
    while (line := file.readline().rstrip()):
        goals.append(tuple(map(float, line.rstrip().split(' '))))

if args.current_robot_num is None:
    print("Please set argument '--current-robot-num'")
    parser.print_help()
    exit()
if args.robots_num is None:
    print("Please set argument '--robot-num'")
    parser.print_help()
    exit()

robot_diameter = args.robot_diameter
robot_radius = args.robot_diameter/2
map_resolution = args.map_resolution

env = StageEnv(current_robot_num=args.current_robot_num,
               robots_num=args.robots_num,
               robot_radius=robot_radius,
               goals=goals,
               map_resolution=args.map_resolution,
               resize_observation=args.resize_observation)

total_reward = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "TB3_MAPF_ENV",
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Agent
# print("o space {}".format(env.observation_space)) # Box(0, 255, (4, 128, 128), uint8)
# print("o space shape {}".format(env.observation_space.shape)) # (4, 128, 128)
# print("o space shape[0] {}".format(env.observation_space.shape[0])) # 4
# print("action space {}".format(env.action_space)) # Box(-1.0, 1.0, (3,), float32)
# print("action space shape {}".format(env.action_space.shape)) # (3,)
agent = SAC(env.observation_space.shape, env.action_space, args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

# for i_episode in range(MAX_EPISODES):
for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        # env.render() ## It costs time
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                # critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                updates += 1

        next_state, reward, done, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        # memory.push(state, action, reward, next_state, done)
        memory.push(state, action, reward, next_state, mask)

        state = next_state

    if total_numsteps > args.num_steps:
        print("### BREAK ###")
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
