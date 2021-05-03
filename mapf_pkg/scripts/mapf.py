#!/usr/bin/env python3
from mapf_env_node import StageEnv
from signal import signal, SIGINT
import argparse
import torch
import random
from gym import spaces

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
goals = [(2, 2, 0), (-1.5, 1.5, 0), (-1.5, -1.5, 0), (2, -2, 0)]

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

for i_episode in range(MAX_EPISODES):

    done = False
    t = 0
    o = env.reset()
    ep_reward = 0

    if i_episode%10==0: print("Start episode: {}".format(i_episode))
    if i_episode%10==0: print("observation size is: {}, type is: {}".format(o.size(), type(o)))

    while not done:
        env.render() ## It costs time

        a = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))

        o, r, done, info = env.step(a)
        # print("Agent take {} and get {} {} {}".format(a, r, done, info))

        if info is not None:
            print(info)

        if done or t >= MAX_EP_STEPS:
            break

env.close()
