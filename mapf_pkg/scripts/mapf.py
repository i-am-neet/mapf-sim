#!/usr/bin/env python3
from mapf_env_node import StageEnv
from signal import signal, SIGINT
import argparse
import torch
import random
import numpy as np
import utils

arg_fmt = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description='Reinforcement Learning Arguments',
                                 formatter_class=arg_fmt)
parser.add_argument('--current-robot-num', type=int,
                    help='The number of current robot. (For env)'
                    )
parser.add_argument('--robots-num', type=int,
                    help='The amount of all robots. (For env)'
                    )
parser.add_argument('--robot-diameter', default=0.25, type=float,
                    help='The diameter of robot (default is according to TB3) (For env)'
                    )
parser.add_argument('--map-resolution', default=0.01, type=float,
                    help='The resolution of map (For env)'
                    )
args = parser.parse_args()

## Tmp
goals = [(2.1, 2.1, 0), (-1.8, 1.8, 3.14), (1.5, -1.5, 3.14), (-2.1, -2.1, 0)]

## RL Args
MAX_EPISODES = 200
MAX_EP_STEPS = 200

def exit_handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

signal(SIGINT, exit_handler)

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
                map_resolution=args.map_resolution)

print(env.observation_space)

total_reward = 0

for i_episode in range(MAX_EPISODES):

    done = False
    t = 0
    o = env.reset()

    if i_episode%10==0: print("Start episode: {}".format(i_episode))
    if i_episode%10==0: print("observation size is: {}, type is: {}".format(o.size(), type(o)))

    while not done:
        # env.render() ## It costs time

        # a = env.action_space.sample()
        a = np.clip([env.current_goal_x - env.current_robot_x, env.current_goal_y - env.current_robot_y, env.current_goal_yaw - env.current_robot_yaw], -0.1, 0.1)
        o, r, done, info = env.step(a)
        # print("Agent take {} and get {} {} {}".format(a, r, done, info))

        total_reward += r

        print("EP {}, Step {}: reward | {} total_reward | {}".format(i_episode, t, r, total_reward))

        if info:
            print(info)

        if done or t >= MAX_EP_STEPS:
            break

        t += 1

env.close()
