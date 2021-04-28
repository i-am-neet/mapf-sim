#!/usr/bin/env python3
import math
import numpy as np
import rospy
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty as EmptySrv
import cv2
import utils
from matplotlib import pyplot as plt

import torch

class StageEnv:

    def __init__(self, current_robot_num,
                       robots_num,
                       robot_radius,
                       goals,
                       map_resolution):

        if len(goals) != robots_num:
            raise ValueError("The amount of goals '%d' must equal to robots_num '%d" %(len(goals), robots_num))

        self.goals = goals
        self.current_robot_num = current_robot_num
        self.robots_num = robots_num
        self.robot_radius = robot_radius
        self.map_resolution = map_resolution

        self.action_space = None
        self.observation_space = None

        ## Initialize tensors
        self.local_map = torch.zeros(1)
        self.agents_map = torch.zeros(1)
        self.my_goal_map = torch.zeros(1)
        self.neighbors_goal_map = torch.zeros(1)

        rospy.init_node('mapf_env_node', anonymous=True)

        rospy.wait_for_service('/reset_positions')
        self._reset_env = rospy.ServiceProxy('/reset_positions', EmptySrv)

        ## Register all topics for message_filters
        subscribers = []
        sub = message_filters.Subscriber("/R"+str(self.current_robot_num)+"_move_base/local_costmap/costmap", OccupancyGrid)
        print("/R"+str(self.current_robot_num)+"_move_base/local_costmap/costmap")
        subscribers.append(sub)
        for i in range(1, self.robots_num+1):
            sub = message_filters.Subscriber("/stage/R"+str(i)+"/odometry", Odometry)
            print("/stage/R"+str(i)+"/odometry")
            subscribers.append(sub)

        ts = message_filters.TimeSynchronizer(subscribers, 10)
        ts.registerCallback(self.__callback)

        rospy.spin()

    """
        callback value:
            data[0]: local_costmap
            data[1]: robot1's odom
            data[2]: robot2's odom
            ...
    """
    def __callback(self, *data):
        ## local costmap: tuple -> np.array -> tensor
        self.map_width = data[0].info.width
        self.map_height = data[0].info.height
        self.local_map = torch.from_numpy(np.asarray(data[0].data).reshape(self.map_height, self.map_width)[::-1].reshape(-1))

        ## Current robot's info
        my_x = data[int(self.current_robot_num)].pose.pose.position.x / self.map_resolution
        my_y = data[int(self.current_robot_num)].pose.pose.position.y / self.map_resolution

        ## Initial size equal to local costmap
        self.my_goal_map = torch.zeros(self.local_map.size())
        self.agents_map = torch.zeros(self.local_map.size())
        self.neighbors_goal_map = torch.zeros(self.local_map.size())

        agx = self.goals[int(self.current_robot_num)-1][0] / self.map_resolution
        agy = self.goals[int(self.current_robot_num)-1][1] / self.map_resolution

        self.my_goal_map = utils.draw_goal(self.my_goal_map, self.map_width, self.map_height, agx - my_x, agy - my_y, self.robot_radius, self.map_resolution)

        for i, e in enumerate(data):
            if i != 0: # data[0] is local costmap
                _rx = e.pose.pose.position.x / self.map_resolution
                _ry = e.pose.pose.position.y / self.map_resolution
                if abs(_rx - my_x) <= self.map_width/2 and abs(_ry - my_y) <= self.map_height/2:

                    self.agents_map = utils.draw_robot(self.agents_map, self.map_width, self.map_height, _rx - my_x, _ry - my_y, self.robot_radius, self.map_resolution)

                    ## Neighbors
                    if i != int(self.current_robot_num):
                        _ngx = self.goals[i-1][0] / self.map_resolution
                        _ngy = self.goals[i-1][1] / self.map_resolution
                        self.neighbors_goal_map = utils.draw_neighbors_goal(self.neighbors_goal_map, self.map_width, self.map_height, _ngx, _ngy, my_x, my_y, self.robot_radius, self.map_resolution)

        # print("check size {} {} {} {}".format(self.local_map.size(), self.my_goal_map.size(), self.agents_map.size(), self.neighbors_goal_map.size()))

    def render(self):

        _im_local_map = utils.tensor_to_cv(self.local_map, self.map_height, self.map_width)
        _im_agents_map = utils.tensor_to_cv(self.agents_map, self.map_height, self.map_width)
        _im_my_goal_map = utils.tensor_to_cv(self.my_goal_map, self.map_height, self.map_width)
        _im_neighbors_goal_map = utils.tensor_to_cv(self.neighbors_goal_map, self.map_height, self.map_width)

        _im_tile = utils.concat_tile_resize([[_im_local_map, _im_agents_map],
                                             [_im_my_goal_map, _im_neighbors_goal_map]],
                                             text=[["local_map", "agents_map"],
                                                   ["my_goal_map", "neighbors_goal_map"]])
        cv2.imshow("Observation", _im_tile)
        cv2.waitKey(1)

    def reset(self):
        self._reset_env()

    def step(self, u):
        pass

    def _get_obs(self):
        pass

    def close(self):
        pass

    def _reward(self):
        pass

    def __del__(self):
        cv2.destroyAllWindows()
