#!/usr/bin/env python3
import math
import numpy as np
import rospy
import rosnode
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty as EmptySrv
from geometry_msgs.msg import Twist
from stage_ros.msg import Stall
from tf.transformations import euler_from_quaternion
import cv2
import utils
from matplotlib import pyplot as plt
import time
import re

import torch

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY(meter)
ARRIVED_RANGE_YAW = math.radians(5) # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW(degrees to radians)

class StageEnv:

    def __init__(self, current_robot_num,
                       robots_num,
                       robot_radius,
                       goals,
                       map_resolution):

        if len(goals) != robots_num:
            raise ValueError("The amount of goals '%d' must equal to robots_num '%d" %(len(goals), robots_num))

        # Initialize variables of Environment
        self.goals = goals
        self.current_robot_num = current_robot_num
        self.robots_num = robots_num
        self.robot_radius = robot_radius
        self.map_resolution = map_resolution
        self.map_height = 0
        self.map_width = 0
        self.current_robot_x = 0
        self.current_robot_y = 0
        self.current_robot_yaw = 0
        self.current_goal_x = 0
        self.current_goal_y = 0
        self.current_goal_yaw = 0

        # Initialize tensors of Observations
        self.local_map = torch.zeros(1)
        self.agents_map = torch.zeros(1)
        self.my_goal_map = torch.zeros(1)
        self.neighbors_goal_map = torch.zeros(1)
        self.observation = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map))

        self.observation_space = self.observation.size()
        self.action_space = len((0, 0, 0)) # vx, vy, w

        self._stalled_robots = tuple()

        rospy.init_node('mapf_env_node', anonymous=True)

        rospy.wait_for_service('/reset_positions')
        self._reset_env = rospy.ServiceProxy('/reset_positions', EmptySrv)

        # Register all topics for message_filters
        _subscribers = []
        _sub_obs = message_filters.Subscriber("/R"+str(self.current_robot_num)+"_move_base/local_costmap/costmap", OccupancyGrid)
        print("/R"+str(self.current_robot_num)+"_move_base/local_costmap/costmap")
        _subscribers.append(_sub_obs)
        for i in range(1, self.robots_num+1):
            _sub_obs = message_filters.Subscriber("/stage/R"+str(i)+"/odometry", Odometry)
            print("/stage/R"+str(i)+"/odometry")
            _subscribers.append(_sub_obs)

        ts = message_filters.TimeSynchronizer(_subscribers, 10)
        ts.registerCallback(self.__callback)

        _sub_stall = rospy.Subscriber("/stalled_robots", Stall, self.__stalled_callback)

        # Flags
        self._sync_ready = False
        self._terminal = False

        # Publisher
        self._pub_vel = rospy.Publisher('/stage/R{}/cmd_vel'.format(self.current_robot_num), Twist, queue_size=1)

        while list(self.observation_space)[1] == 1:
            print("waiting ROS message_filters...")
            time.sleep(1)

        # Waiting until all mapf_env_node started
        rosnode.rosnode_cleanup()
        _check_regex = re.compile("/mapf_env_node*")
        MAPF_ALIVE_NODES = list()
        while len(MAPF_ALIVE_NODES) != self.robots_num:
            print("waiting all mapf_env_node...")
            rosnode.rosnode_cleanup()
            MAPF_ALIVE_NODES.clear()
            MAPF_NODES = list(filter(_check_regex.match, rosnode.get_node_names()))
            for n in MAPF_NODES:
                if rosnode.rosnode_ping(n, max_count=3):
                    MAPF_ALIVE_NODES.append(n)
            time.sleep(1)


        # rospy.spin()

    def __callback(self, *data):
        """
        callback value:
            data[0]: local_costmap
            data[1]: robot1's odom
            data[2]: robot2's odom
            ...

        Observations are created by this function
        (local_map, agents_map, my_goal_map, neighbors_goal_map)

        """
        # local costmap: tuple -> np.array -> tensor
        self.map_width = data[0].info.width
        self.map_height = data[0].info.height
        self.local_map = torch.from_numpy(np.asarray(data[0].data).reshape(self.map_height, self.map_width)[::-1].reshape(-1))

        # Current robot's info
        self.current_robot_x = data[int(self.current_robot_num)].pose.pose.position.x
        self.current_robot_y = data[int(self.current_robot_num)].pose.pose.position.y
        
        self.current_robot_yaw = euler_from_quaternion([data[int(self.current_robot_num)].pose.pose.orientation.x,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.y,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.z,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.w])[2]

        my_x = self.current_robot_x / self.map_resolution
        my_y = self.current_robot_y / self.map_resolution
        my_yaw = self.current_robot_yaw

        # Initialize size equal to local costmap
        self.my_goal_map = torch.zeros(self.local_map.size())
        self.agents_map = torch.zeros(self.local_map.size())
        self.neighbors_goal_map = torch.zeros(self.local_map.size())

        self.current_goal_x = self.goals[int(self.current_robot_num)-1][0]    # Agent's goal x
        self.current_goal_y = self.goals[int(self.current_robot_num)-1][1]    # Agent's goal y
        self.current_goal_yaw = self.goals[int(self.current_robot_num)-1][2]  # Agent's goal yaw

        agx = self.current_goal_x / self.map_resolution
        agy = self.current_goal_y / self.map_resolution
        agyaw = self.current_goal_yaw

        self.my_goal_map = utils.draw_goal(self.my_goal_map, self.map_width, self.map_height, agx - my_x, agy - my_y, self.robot_radius, self.map_resolution)

        for i, e in enumerate(data):
            if i != 0: # data[0] is local costmap
                _rx = e.pose.pose.position.x / self.map_resolution
                _ry = e.pose.pose.position.y / self.map_resolution
                if abs(_rx - my_x) <= self.map_width/2 and abs(_ry - my_y) <= self.map_height/2:

                    self.agents_map = utils.draw_robot(self.agents_map, self.map_width, self.map_height, _rx - my_x, _ry - my_y, self.robot_radius, self.map_resolution)

                    # Neighbors
                    if i != int(self.current_robot_num):
                        _ngx = self.goals[i-1][0] / self.map_resolution     # Neighbor's goal x
                        _ngy = self.goals[i-1][1] / self.map_resolution     # Neighbor's goal y
                        _ngyaw = self.goals[i-1][2]                         # Neighbor's goal yaw
                        self.neighbors_goal_map = utils.draw_neighbors_goal(self.neighbors_goal_map, self.map_width, self.map_height, _ngx, _ngy, my_x, my_y, self.robot_radius, self.map_resolution)

        # print("check size {} {} {} {}".format(self.local_map.size(), self.my_goal_map.size(), self.agents_map.size(), self.neighbors_goal_map.size()))
        self.observation = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map))
        self.observation_space = self.observation.size()

        self._sync_ready = True

    def __stalled_callback(self, data):
        """
        Get stalled robots' info from stage.
        NOTICE: robot's number is count from 0
        """

        self._stalled_robots = tuple(i+1 for i in data.stalled_robot_num)

        if len(data.stalled_robot_num) > 0:
            self._terminal = True

    def render(self):
        """
        Show the observation that contains local_map, agents_map, my_goal_map, and neighbors_goal_map
        """

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
        """
        Reset robots' position.
        Output: observation
        """

        self._sync_ready = False
        self._terminal = False

        self._reset_env() # call ROS service
        return self.observation

    def step(self, u):
        """
        Input: action
        Output: observation, reward, done, info
        """

        self._sync_ready = False

        self.__action_to_vel(u)

        # Reward function
        r = 0

        # ARRIVED GOAL
        if math.dist([self.current_robot_x, self.current_robot_y],
                     [self.current_goal_x, self.current_goal_y]) <= ARRIVED_RANGE_XY and \
           abs(self.current_robot_yaw - self.current_goal_yaw) <= ARRIVED_RANGE_YAW:

           print("Arrived!!!")
           r = 100
        else:
            print("Ddist: {}, Dyaw: {}".format(math.dist([self.current_robot_x, self.current_robot_y], [self.current_goal_x, self.current_goal_y]),
                                               abs(self.current_robot_yaw - self.current_goal_yaw)))

        # The robot which is in collision will get punishment
        if self.current_robot_num in self._stalled_robots:
            r = -100
            return self.observation, r, True, {"Somebody screw up: {}".format(self._stalled_robots)}

        while not self._sync_ready:
            pass

        return self.observation, r, False, {}

    def __action_to_vel(self, action):
        if len(action) != self.action_space:
            raise ValueError("action size ERROR.")

        msg = Twist()
        msg.linear.x = action[0]
        msg.linear.y = action[1]
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = action[2]
        self._pub_vel.publish(msg)

    def close(self):
        pass

    def __del__(self):
        cv2.destroyAllWindows()
