#!/usr/bin/env python3
import math
import numpy as np
import rospy
import rosnode
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty as EmptySrv
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from gazebo_msgs.msg import ContactsState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv2
import utils
from matplotlib import pyplot as plt
import time
import re
import gym
from gym import spaces
from gym.utils import seeding

import torch

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY (meter)
ARRIVED_RANGE_YAW = math.radians(5) # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW (degrees to radians)

MAX_SPEED = abs(1.0) # maximum speed (m/s)

class StageEnv(gym.Env):

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
        self._current_robot_x = 0
        self._current_robot_y = 0
        self._current_robot_yaw = 0
        self._current_robot_init_x = 0
        self._current_robot_init_y = 0
        self._current_robot_init_yaw = 0
        self._current_robot_done = False
        self._first_time = True
        self._current_goal_x = 0
        self._current_goal_y = 0
        self._current_goal_yaw = 0
        self.robots_position = list()
        self._collision = False

        # Initialize tensors of Observations
        self.local_map = torch.zeros(1)
        self.agents_map = torch.zeros(1)
        self.my_goal_map = torch.zeros(1)
        self.neighbors_goal_map = torch.zeros(1)
        self.observation = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map))

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.map_height, self.map_width, 4), dtype=np.uint8)
        self.action_space = spaces.Box(low=-MAX_SPEED, high=MAX_SPEED, shape=(3,), dtype=np.float32)

        self._done_robots = tuple()

        rospy.init_node('mapf_env_node', anonymous=True)

        # Publisher
        self._pub_vel = rospy.Publisher('/robot{}/mobile/cmd_vel'.format(self.current_robot_num), Twist, queue_size=1)
        # self._pub_pose = rospy.Publisher('/stage/robot_{}/cmd_pose'.format(self.current_robot_num), Pose, queue_size=1)
        self._pub_done = rospy.Publisher('/robot{}/done'.format(self.current_robot_num), Bool, queue_size=1)

        # Subscriber
        # Register all topics for message_filters
        _subscribers = []
        _sub_obs = message_filters.Subscriber("/robot{}_move_base/local_costmap/costmap".format(str(self.current_robot_num)), OccupancyGrid)
        print("/robot{}_move_base/local_costmap/costmap".format(str(self.current_robot_num)))
        _subscribers.append(_sub_obs)
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}/mobile/odom".format(str(i)), Odometry)
            print("/robot{}/mobile/odom".format(str(i)))
            _subscribers.append(_sub_obs)

        # ts = message_filters.TimeSynchronizer(_subscribers, 10)
        ts = message_filters.ApproximateTimeSynchronizer(_subscribers, 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.__callback)

        _sub_collision = rospy.Subscriber("/robot{}/bumper".format(str(self.current_robot_num)), ContactsState, self.__collision_callback)

        # Flags
        self._sync_obs_ready = False

        _subscribers_done = []
        for i in range(0, self.robots_num):
            _sub_done = message_filters.Subscriber("/robot{}/done".format(str(i)), Bool)
            print("/robot{}/done".format(str(i)))
            _subscribers_done.append(_sub_done)

        ts_done = message_filters.ApproximateTimeSynchronizer(_subscribers_done, 10, 0.1, allow_headerless=True)
        ts_done.registerCallback(self.__callback_done)

        while self.map_height == 0:
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

        data = data[1:] # Remove local_costmap, for counting robot's number easily

        # Current robot's info
        self._current_robot_x = data[int(self.current_robot_num)].pose.pose.position.x
        self._current_robot_y = data[int(self.current_robot_num)].pose.pose.position.y
        self._current_robot_yaw = euler_from_quaternion([data[int(self.current_robot_num)].pose.pose.orientation.x,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.y,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.z,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.w])[2]

        if self._first_time:
            # Save initial position
            self._current_robot_init_x = self._current_robot_x
            self._current_robot_init_y =  self._current_robot_y
            self._current_robot_init_yaw =  self._current_robot_yaw
            self._current_goal_x = self.goals[int(self.current_robot_num)][0]    # Agent's goal x
            self._current_goal_y = self.goals[int(self.current_robot_num)][1]    # Agent's goal y
            self._current_goal_yaw = self.goals[int(self.current_robot_num)][2]  # Agent's goal yaw

            print("I am robot {}, from ({}, {}, {}) to ({}, {}, {})"\
                   .format(self.current_robot_num,\
                           self._current_robot_init_x, self._current_robot_init_y, self._current_robot_init_yaw,\
                           self._current_goal_x, self._current_goal_y, self._current_goal_yaw))
            self._first_time = False

        # Scale unit for pixel with map's resolution (meter -> pixels)
        my_x = self._current_robot_x / self.map_resolution
        my_y = self._current_robot_y / self.map_resolution
        my_yaw = self._current_robot_yaw

        agx = self._current_goal_x / self.map_resolution
        agy = self._current_goal_y / self.map_resolution
        agyaw = self._current_goal_yaw

        # Initialize size equal to local costmap
        self.my_goal_map = torch.zeros(self.local_map.size())
        self.agents_map = torch.zeros(self.local_map.size())
        self.neighbors_goal_map = torch.zeros(self.local_map.size())

        self.my_goal_map = utils.draw_goal(self.my_goal_map, self.map_width, self.map_height, agx - my_x, agy - my_y, self.robot_radius, self.map_resolution)

        self.robots_position = list()

        for i, e in enumerate(data):
            self.robots_position.append([e.pose.pose.position.x,
                                         e.pose.pose.position.y,
                                         euler_from_quaternion([e.pose.pose.orientation.x,
                                                                e.pose.pose.orientation.y,
                                                                e.pose.pose.orientation.z,
                                                                e.pose.pose.orientation.w])[2]])
            _rx = e.pose.pose.position.x / self.map_resolution
            _ry = e.pose.pose.position.y / self.map_resolution

            if abs(_rx - my_x) <= self.map_width/2 and abs(_ry - my_y) <= self.map_height/2:

                self.agents_map = utils.draw_robot(self.agents_map, self.map_width, self.map_height, _rx - my_x, _ry - my_y, self.robot_radius, self.map_resolution)

                # Neighbors
                if i != int(self.current_robot_num):
                    _ngx = self.goals[i][0] / self.map_resolution     # Neighbor's goal x
                    _ngy = self.goals[i][1] / self.map_resolution     # Neighbor's goal y
                    _ngyaw = self.goals[i][2]                         # Neighbor's goal yaw
                    self.neighbors_goal_map = utils.draw_neighbors_goal(self.neighbors_goal_map, self.map_width, self.map_height, _ngx, _ngy, my_x, my_y, self.robot_radius, self.map_resolution)

        # print("check size {} {} {} {}".format(self.local_map.size(), self.my_goal_map.size(), self.agents_map.size(), self.neighbors_goal_map.size()))
        self.observation = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map))
        # self.observation_space = self.observation.size()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.map_height, self.map_width, 4), dtype=np.uint8)

        self._sync_obs_ready = True

    def __callback_done(self, *data):
        """
        callback value:
            data[0]: whether robot0 arrived it's goal
            data[1]: whether robot1 arrived it's goal
            data[2]: whether robot2 arrived it's goal
            ...
        """

        self._done_robots = tuple(e.data for i, e in enumerate(data))

    @property
    def done_robots(self):
        return self._done_robots

    @property
    def all_robots_done(self):
        if len(self.done_robots) == 0:
            return False
        else:
            return all(self._done_robots)

    @property
    def i_am_done(self):
        return self._current_robot_done

    @i_am_done.setter
    def i_am_done(self, done):
        self._current_robot_done = done

    def __collision_callback(self, data):
        """
        Get model contact status by gazebo bumper plugin.
        NOTICE: In this data, gets contact pair between models.
        """
        self._collision = False
        # print("Collision Length: {}".format(len(data.states)))
        for i, e in enumerate(data.states):
            # print("Pair {}: {} <---> {}".format(i, e.collision1_name, e.collision2_name))
            A = [e.collision1_name, e.collision2_name]

            if any('ground_plane' in a.lower() for a in A):
                # Ignore collision with ground_plane
                break
            elif any('wall' in a.lower() for a in A):
                # print("{} Hit the wall!!!!!".format(c))
                self._collision = True
                break
            elif any('door' in a.lower() for a in A):
                # print("{} Hit the wall!!!!!".format(c))
                self._collision = True
                break
            elif all('robot' in a.lower() for a in A):
                # print("{} Hit other robot!!".format(c))
                self._collision = True
                break
            else:
                raise Exception("Unknown collision condition, collision pair:\n {} <---> {}".format(A[0], A[1]))
        

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
        cv2.imshow("Observation of Agent {}".format(self.current_robot_num), _im_tile)
        cv2.waitKey(1)

    def reset(self):
        """
        When all robots are done!
        Reset robots' position, robots' done.

        Output: observation
        """

        self._current_robot_done = False
        self._done_robots = tuple()
        self._sync_obs_ready = False

        self.__reset_all_robots()

        return self.observation

    def step(self, u):
        """
        Input: action
        Output: observation, reward, done, info
        """

        done = False
        info = {}
        self._sync_obs_ready = False
        self._current_robot_done = False

        if self._current_robot_done:
            self.__action_to_vel([0, 0, 0])
        else:
            self.__action_to_vel(u)

        # moving cost
        r = -1

        # ARRIVED GOAL
        for i, e in enumerate(self.robots_position):
            if utils.dist([e[0], e[1]], [self.goals[i][0], self.goals[i][1]]) <= ARRIVED_RANGE_XY and \
               abs(e[2] - self.goals[i][2]) <= ARRIVED_RANGE_YAW:

                if i == self.current_robot_num:
                    self._current_robot_done = True
                    r = 100

        ## The robot which is in collision will get punishment
        if self._collision:
            self._current_robot_done = True
            r = -10
            info = {"I got collision..."}

        self._pub_done.publish(self._current_robot_done)

        while not self._sync_obs_ready:
            pass

        if self.all_robots_done:
            done = True

        return self.observation, r, done, info

    def __action_to_vel(self, action):
        if action.shape != self.action_space.shape:
            raise ValueError("action size ERROR.")

        msg = Twist()
        msg.linear.x = action[0]
        msg.linear.y = action[1]
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = action[2]
        self._pub_vel.publish(msg)

    # TODO: Change to Gazebo
    def __reset_current_robot(self):
        """
        Reset current robot's position to _current_robot_init_(x, y, yaw)
        """

        # Need to stop robot for a while before resetting pose!!!!!
        self.__action_to_vel([0, 0, 0])
        time.sleep(3)

        init_pose = Pose()
        init_pose.position.x = self._current_robot_init_x
        init_pose.position.y = self._current_robot_init_y
        init_pose.position.z = 0
        init_pose.orientation.x = quaternion_from_euler(0, 0, self._current_robot_init_yaw)[0]
        init_pose.orientation.y = quaternion_from_euler(0, 0, self._current_robot_init_yaw)[1]
        init_pose.orientation.z = quaternion_from_euler(0, 0, self._current_robot_init_yaw)[2]
        init_pose.orientation.w = quaternion_from_euler(0, 0, self._current_robot_init_yaw)[3]

        # self._pub_pose.publish(init_pose)

    def __stop_all_robots(self):
        """
        Publish cmd_vel: (0, 0, 0) to all robots
        """
        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        for i in range(0, self.robots_num):
            _pub = rospy.Publisher('/stage/robot_{}/cmd_vel'.format(i), Twist, queue_size=1)
            _pub.publish(msg)

    def __reset_all_robots(self):
        """
        # Reset all robots' position by robot 0
        Reset all robots' position
        """

        ## stop robots for Stage
        # self.__stop_all_robots()
        # time.sleep(3)

        # if self.current_robot_num == 0:
        #     print("I am robot 0, I reset Env.")
        try:
            # reset_env = rospy.ServiceProxy('/reset_positions', EmptySrv)
            reset_env = rospy.ServiceProxy('/gazebo/reset_world', EmptySrv)
            reset_env()
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

    def close(self):
        pass

    def __del__(self):
        # Stop robot
        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0
        self._pub_vel.publish(msg)
        # Close cv
        cv2.destroyAllWindows()
