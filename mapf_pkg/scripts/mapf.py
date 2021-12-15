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
parser.add_argument('--env_name', default="TB3_MAPF_ENV",
                    help='name of the environment to run')
parser.add_argument('-crn', '--current_robot_num', type=int,
                    help='The number of current robot. (For env)'
                    )
parser.add_argument('-rn', '--robots_num', type=int,
                    help='The amount of all robots. (For env)'
                    )
parser.add_argument('--robot_diameter', default=0.25, type=float,
                    help='The diameter of robot (default is according to TB3) (For env)'
                    )
parser.add_argument('--map_resolution', default=0.05, type=float,
                    help='The resolution of map (For env)'
                    )
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                    help='learning rate (default: 0.001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--max_episode_steps', type=int, default=200, metavar='N',
                    help='maximum steps of episode (default: 200)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=100, metavar='N',
                    help='model updates per simulator step (default: 100)')
parser.add_argument('--expert_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling expert actions (default: 10000)')
parser.add_argument('--start_steps', type=int, default=20000, metavar='N',
                    help='Steps sampling random actions (default: 20000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--goals_init_file', type=str, default=os.environ.get('HOME')+"/mapf_ws/src/gazebo/robot_gazebo/config/init_goals_one.cfg",
                    help="config file of goals' init info (default: '$HOME/mapf_ws/src/gazebo/robot_gazebo/config/init_goals_one.cfg')")
parser.add_argument('--poses_init_file', type=str, default=os.environ.get('HOME')+"/mapf_ws/src/gazebo/robot_gazebo/config/init_poses_one.cfg",
                    help="config file of poses' init info (default: '$HOME/mapf_ws/src/gazebo/robot_gazebo/config/init_poses_one.cfg')")
# parser.add_argument('--resize_observation', type=int, default=64, metavar='N',
#                     help='Resize observation size (default: 64)')
parser.add_argument('--load_model', type=bool, default=False,
                    help='Use pretrain model (default: False)')
parser.add_argument('--use_expert', type=bool, default=False,
                    help='Use expert (default: False)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
parser.add_argument('--note', type=str, default="",
                    help="The note string will be added on log's file name")
parser.add_argument('--render', type=bool, default=False,
                    help='Show env render (default: False)')
parser.add_argument('--test', type=bool, default=False,
                    help='Testing, will not record anything (default: False)')

args = parser.parse_args()

init_poses = []
goals = []

with open(args.poses_init_file) as file:
    while (line := file.readline().rstrip()):
        init_poses.append(tuple(map(float, line.rstrip().split(' '))))

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
               init_poses=init_poses,
               goals=goals,
               map_resolution=args.map_resolution)

total_reward = 0

torch.manual_seed(args.seed)
np.random.seed(args.seed)

#Tesnorboard
if not args.test:
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "use_expert_" if args.use_expert else "",
                                                             args.note))
else:
    print("*****************************************")
    print("     TESTING, WON'T RECORD ANYTHING!     ")
    print("*****************************************")

# Agent
agent = SAC(env.observation_space, env.action_space, args)

if args.load_model is True and os.path.exists('models/'):
    agent.load_model("models/sac_actor_{}_".format(args.env_name),
                     "models/sac_critic_{}_".format(args.env_name))
else:
    print("Does not use pretrain model.")

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    plan_len = env.planner_benchmark

    for _ in range(args.max_episode_steps):

        if args.render:
            env.render() ## It costs time

        if len(memory) > args.batch_size:

            if i_episode % 10 == 0:
                print("Start Update")
                # It cost times, does it need to stop robots?
                env.stop_robots()

                # Number of updates per step in environment
                for _ in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    if not args.test:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                    # # SAC_V
                    # value_loss, critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)

                    # writer.add_scalar('loss/value', value_loss, updates)
                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    updates += 1
                agent.lr_decay_step()
                break

        if args.expert_steps > total_numsteps and args.use_expert:
            action = env.expert_action()
        elif args.start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        _expert_action = env.expert_action()
        next_state, reward, done, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # print("episode steps: {}, reward: {}".format(episode_steps, round(episode_reward, 2)))

        # Ignore the "done" signal if it comes from hitting the time horizon
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        # memory.push(state, action, reward, next_state, mask)
        memory.push(state, action, reward, next_state, done)
        memory.push_expert(state, _expert_action)

        state = next_state

        if done:
            break

    if total_numsteps > args.num_steps:
        print("### BREAK ###")
        break

    if not args.test:
        writer.add_scalar('reward/train', episode_reward/plan_len, i_episode)
    print("---")
    print("Episode: {}, total numsteps: {}, episode steps: {}".format(i_episode, total_numsteps, episode_steps))
    print("reward {} / planner length {} = {}".format(round(episode_reward, 2), plan_len, round(episode_reward/plan_len, 2)))
    print("---")

    if i_episode % 50 == 0 and args.eval is True:
        print("Start Testing")
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            plan_len = env.planner_benchmark
            # while not done:
            for t in range(args.max_episode_steps):
                action = agent.select_action(state, eval=True)

                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # memory.push(state, action, reward, next_state, done)

                state = next_state

                if t % 20 == 0:
                    print(t)

                if done:
                    break

            avg_reward += episode_reward/plan_len
        avg_reward /= episodes

        if not args.test:
            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            agent.save_model(args.env_name)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
