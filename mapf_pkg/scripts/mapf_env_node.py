#!/usr/bin/env python3
import math
import numpy as np
import cv2
import utils
import time
import re
import gym
from gym import spaces
from gym.utils import seeding
from signal import signal, SIGINT
from mapf_ros_node import MyRosBridge

import torch
from torchvision import transforms

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY (meter)
ARRIVED_RANGE_YAW = math.radians(5)   # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW (degrees to radians)

MAX_SPEED = abs(0.5) # maximum speed (m/s)

class StageEnv(gym.Env):

    def __init__(self, current_robot_num,
                       robots_num,
                       robot_radius,
                       init_poses,
                       goals,
                       map_resolution,
                       resize_observation=0):

        if len(goals) != robots_num:
            raise ValueError("The amount of goals '%d' must equal to robots_num '%d" %(len(goals), robots_num))

        if len(init_poses) != robots_num:
            raise ValueError("The amount of init_poses '%d' must equal to robots_num '%d" %(len(init_poses), robots_num))

        signal(SIGINT, self.exit_handler)

        self.ros = MyRosBridge(current_robot_num, robots_num, map_resolution, goals, robot_radius)

        # Initialize ROS environment
        self.ros.stop_robots()
        self.ros.clear_map()
        self.ros.reset_poses(init_poses, goals)
        self.ros.movebase_client(goals[current_robot_num][0], goals[current_robot_num][1], goals[current_robot_num][2])
        self.ros.get_observation

        # Initialize variables of Environment
        self.goals = goals
        self.current_robot_num = current_robot_num
        self.robots_num = robots_num
        self.resize_observation = resize_observation
        self._planner_benchmark = 0

        self.observation_space = spaces.Dict(map=spaces.Box(low=0, high=1, shape=(4, self.ros.map_height, self.ros.map_width), dtype=np.float32),
                                             lidar=spaces.Box(low=0, high=1, shape=(1, self.ros.lidar_range_size), dtype=np.float32),
                                             goal=spaces.Box(low=0, high=math.pi, shape=(1, 2)))
        self.action_space = spaces.Box(low=-MAX_SPEED, high=MAX_SPEED, shape=(3,), dtype=np.float32)


    def render(self):
        """
        Show the observation that contains local_map, agents_map, planner_map, and neighbors_goal_map
        """
        _map_0 = self.ros.get_observation['map'][0].astype('uint8')
        _map_1 = self.ros.get_observation['map'][1].astype('uint8')
        _map_2 = self.ros.get_observation['map'][2].astype('uint8')
        _map_3 = self.ros.get_observation['map'][3].astype('uint8')
        _im_tile = utils.concat_tile_resize([[_map_0, _map_1],
                                             [_map_2, _map_3]],
                                             text=[["local_map", "agents_map"],
                                                 ["planner_map", "neighbors_goal_map"]])
        cv2.imshow("Observation of Agent {}".format(self.current_robot_num), _im_tile)
        cv2.waitKey(1)

    def reset(self):
        """
        When all robots are done!
        Reset robots' position, robots' done.

        Output: observation
        """

        self.ros.stop_robots()
        time.sleep(1)

        self.ros.clear_map()

        _init_poses = self.ros.get_new_poses()
        self.goals = self.ros.get_new_poses()
        self.ros.goals = self.goals
        self.ros.reset_poses(_init_poses, self.goals)
        time.sleep(1)
        self.ros.movebase_client(self.goals[self.current_robot_num][0], self.goals[self.current_robot_num][1], self.goals[self.current_robot_num][2])
        time.sleep(1)

        self._planner_benchmark = len(self.ros.planner_path)

        _o = self.normalize_observation(self.ros.get_observation)
        return _o

    def step(self, u):

        # check move_base is working
        if not self.ros.check_movebase:
            self.reset()

        done = False
        info = {}

        if u.shape != self.action_space.shape:
            raise ValueError("action size ERROR.")

        # planner's direction reward
        _p_x = self.ros.planner_path[0].pose.position.x
        _p_y = self.ros.planner_path[0].pose.position.y
        if any([_p_x, _p_y]) and any([u[0], u[1]]):
            r = (utils.angle_between([_p_x, _p_y], [u[0], u[1]]) / math.pi) * -0.3 - 0.01
        else:
            print("################## OMG ######################")
            r = -0.01

        self.ros.action_to_vel(u)

        # moving cost
        # r = -0.3

        # planner reward
        _path_len = len(self.ros.planner_path)
        r += np.sign(_path_len - self._planner_benchmark) * -0.1
        self._planner_benchmark = _path_len

        # ARRIVED GOAL
        _r_x = self.ros.odom[0]
        _r_y = self.ros.odom[1]
        _r_yaw = self.ros.odom[2]
        _g_x = self.goals[self.current_robot_num][0]
        _g_y = self.goals[self.current_robot_num][1]
        _g_yaw = self.goals[self.current_robot_num][2]
        if utils.dist([_r_x, _r_y], [_g_x, _g_y]) <= ARRIVED_RANGE_XY and abs(_r_yaw - _g_yaw) <= ARRIVED_RANGE_YAW:
            done = True
            r = 50

        # robot's direction reward
        r += np.cos(abs(_g_yaw - _r_yaw)) * 0.1 - 0.1

        ## The robot which is in collision will get punishment
        if self.ros.collision_check:
            r += -2.0
            info = {"I got collision..."}

        _o = self.normalize_observation(self.ros.get_observation)
        return _o, r, done, info

    def close(self):
        self.ros.stop_robots()
        pass

    def __del__(self):
        self.ros.stop_robots()
        # Close cv
        cv2.destroyAllWindows()

    def exit_handler(self, signal_received, frame):
        # Handle any cleanup here
        self.ros.stop_robots()
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        exit(0)

    def expert_action(self):
        e_u = self.ros.expert_action
        return e_u

    def stop_robots(self):
        self.ros.stop_robots()

    def normalize_observation(self, o):
        _o = dict()
        # Normalize map to 0-1
        _o['map'] = o['map']/255
        # Cutting lidar
        _l = np.asarray(o['lidar'][0])
        _l[_l > 1] = 1
        _o['lidar'] = [_l.tolist()]
        # Goal
        _o['goal'] = o['goal']
        return _o
