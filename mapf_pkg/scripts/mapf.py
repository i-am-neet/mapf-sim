#!/usr/bin/env python3
from mapf_env_node import StageEnv
import argparse
from signal import signal, SIGINT

arg_fmt = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(description='Stage RL Env Args',
                                    formatter_class=arg_fmt)
parser.add_argument('--current-robot-num', type=int,
                    help='The number of current robot.'
                    )
parser.add_argument('--robots-num', type=int,
                    help='The amount of all robots.'
                    )
parser.add_argument('--robot-diameter', default=0.25, type=float,
                    help='The diameter of robot (default: 0.25, according to TB3)'
                    )
parser.add_argument('--map-resolution', default=0.01, type=float,
                    help='The resolution of map (default: 0.01)'
                    )
args = parser.parse_args()

## Tmp
goals = [(2, 2), (-1.5, 1.5), (-1.5, -1.5), (2, -2)]

def exit_handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

if __name__ == '__main__':

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
