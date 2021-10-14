#!/usr/bin/env python3
import copy
import math
import numpy as np
import rospy
import rosnode
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Empty as EmptySrv
from geometry_msgs.msg import Twist, Pose, Quaternion
from std_msgs.msg import Bool
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv2
import utils
from matplotlib import pyplot as plt
import time
import re
import gym
from gym import spaces
from gym.utils import seeding
from signal import signal, SIGINT
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import torch
from torchvision import transforms

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY (meter)
ARRIVED_RANGE_YAW = math.radians(5) # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW (degrees to radians)

MAX_SPEED = abs(1.0) # maximum speed (m/s)

class StageEnv(gym.Env):

    def __init__(self, current_robot_num,
                       robots_num,
                       robot_radius,
                       init_poses,
                       goals,
                       map_resolution,
                       resize_observation):

        if len(goals) != robots_num:
            raise ValueError("The amount of goals '%d' must equal to robots_num '%d" %(len(goals), robots_num))

        if len(init_poses) != robots_num:
            raise ValueError("The amount of init_poses '%d' must equal to robots_num '%d" %(len(init_poses), robots_num))

        signal(SIGINT, self.exit_handler)

        # Initialize variables of Environment
        self.init_poses = init_poses
        self.goals = goals
        self.current_robot_num = current_robot_num
        self.robots_num = robots_num
        self.robot_radius = robot_radius
        self.map_resolution = map_resolution
        self.map_height = 0
        self.map_width = 0
        self._current_robot_x = init_poses[int(current_robot_num)][0]
        self._current_robot_y = init_poses[int(current_robot_num)][1]
        self._current_robot_yaw = init_poses[int(current_robot_num)][2]
        self._current_robot_done = False
        self._current_goal_x = goals[int(current_robot_num)][0]    # Agent's goal x
        self._current_goal_y = goals[int(current_robot_num)][1]    # Agent's goal y
        self._current_goal_yaw = goals[int(current_robot_num)][2]  # Agent's goal yaw
        self.robots_position = list()
        self._collision = False
        self._max_episode_steps = 200
        self.resize_observation = resize_observation

        # Initialize tensors of Observations
        self.local_map = torch.zeros(1)
        self.agents_map = torch.zeros(1)
        # self.my_goal_map = torch.zeros(1)
        self.planner_map = torch.zeros(1)
        self.neighbors_goal_map = torch.zeros(1)
        # self.observation = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map)).numpy()
        self.observation = torch.stack((self.local_map, self.agents_map, self.planner_map, self.neighbors_goal_map)).numpy()

        self.observation_space = spaces.Box(low=0, high=255, shape=(4, self.map_height, self.map_width), dtype=np.uint8)
        self.action_space = spaces.Box(low=-MAX_SPEED, high=MAX_SPEED, shape=(3,), dtype=np.float32)

        self._done_robots = tuple()

        rospy.init_node('mapf_env_node', anonymous=True)

        # Publisher
        self._pub_vel = rospy.Publisher('/robot{}/mobile/cmd_vel'.format(self.current_robot_num), Twist, queue_size=1)
        self._pub_done = rospy.Publisher('/robot{}/done'.format(self.current_robot_num), Bool, queue_size=1)

        # Subscriber
        # Register all topics for message_filters
        _subscribers = []
        _sub_obs = message_filters.Subscriber("/robot{}_move_base/local_costmap/costmap".format(str(self.current_robot_num)), OccupancyGrid)
        print("/robot{}_move_base/local_costmap/costmap".format(str(self.current_robot_num)))
        _subscribers.append(_sub_obs)
        _sub_obs = message_filters.Subscriber("/robot{}_move_base/NavfnROS/plan".format(str(self.current_robot_num)), Path)
        print("/robot{}_move_base/NavfnROS/plan".format(str(self.current_robot_num)))
        _subscribers.append(_sub_obs)
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}/mobile/odom".format(str(i)), Odometry)
            print("/robot{}/mobile/odom".format(str(i)))
            _subscribers.append(_sub_obs)

        # ts = message_filters.TimeSynchronizer(_subscribers, 10)
        ts = message_filters.ApproximateTimeSynchronizer(_subscribers, 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.__callback)

        _sub_collision = rospy.Subscriber("/robot{}/bumper".format(str(self.current_robot_num)), ContactsState, self.__collision_callback)

        # Action
        self.__movebase_client(self.current_robot_num, self._current_goal_x, self._current_goal_y, self._current_goal_yaw)

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

        # Save costmap
        self.global_costmap = self.__get_global_costmap(self.current_robot_num)

        # rospy.spin()

    def __callback(self, *data):
        """
        callback value:
            data[0]: local_costmap
            data[1]: path
            data[2]: robot1's odom
            data[3]: robot2's odom
            ...

        Observations are created by this function
        (local_map, agents_map, planner_map, neighbors_goal_map)

        """
        # local costmap: tuple -> np.array -> tensor
        self.map_width = data[0].info.width
        self.map_height = data[0].info.height
        # print("0. {}".format(torch.from_numpy(np.asarray(data[0].data)).shape))
        # print("1. {}".format(torch.from_numpy(np.asarray(data[0].data).reshape(self.map_height, self.map_width)).shape))
        # print("2. {}".format(torch.from_numpy(np.asarray(data[0].data).reshape(self.map_height, self.map_width)[::-1].reshape(-1)).shape))
        self.local_map = torch.from_numpy(np.asarray(data[0].data).reshape(self.map_height, self.map_width)[::-1].reshape(-1))

        planner_path = data[1].poses

        data = data[2:] # Remove local_costmap, for counting robot's number easily

        # Current robot's info
        self._current_robot_x = data[int(self.current_robot_num)].pose.pose.position.x
        self._current_robot_y = data[int(self.current_robot_num)].pose.pose.position.y
        self._current_robot_yaw = euler_from_quaternion([data[int(self.current_robot_num)].pose.pose.orientation.x,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.y,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.z,
                                                        data[int(self.current_robot_num)].pose.pose.orientation.w])[2]

        # Scale unit for pixel with map's resolution (meter -> pixels)
        my_x = self._current_robot_x / self.map_resolution
        my_y = self._current_robot_y / self.map_resolution
        my_yaw = self._current_robot_yaw

        agx = self._current_goal_x / self.map_resolution
        agy = self._current_goal_y / self.map_resolution
        agyaw = self._current_goal_yaw

        # Initialize size equal to local costmap
        # self.my_goal_map = torch.zeros(self.local_map.size())
        self.planner_map = torch.zeros(self.local_map.size())
        self.agents_map = torch.zeros(self.local_map.size())
        self.neighbors_goal_map = torch.zeros(self.local_map.size())

        # self.my_goal_map = utils.draw_goal(self.my_goal_map, self.map_width, self.map_height, agx - my_x, agy - my_y, self.robot_radius, self.map_resolution)
        self.planner_map = utils.draw_path(self.planner_map, self.map_width, self.map_height, my_x, my_y, planner_path, self.map_resolution)

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

        # Reshape map tensor to 2-dims
        self.local_map = self.local_map.reshape(self.map_height, self.map_width)
        self.agents_map = self.agents_map.reshape(self.map_height, self.map_width)
        # self.my_goal_map = self.my_goal_map.reshape(self.map_height, self.map_width)
        self.planner_map = self.planner_map.reshape(self.map_height, self.map_width)
        self.neighbors_goal_map = self.neighbors_goal_map.reshape(self.map_height, self.map_width)
        # Observation is stack all map tensor, then convert to np.array
        # o = torch.stack((self.local_map, self.agents_map, self.my_goal_map, self.neighbors_goal_map))
        o = torch.stack((self.local_map, self.agents_map, self.planner_map, self.neighbors_goal_map))
        # Resize map (resize operation on tensor)
        transform =  transforms.Resize(self.resize_observation)
        self.observation = transform(o).numpy()
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, self.resize_observation, self.resize_observation), dtype=np.uint8)

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
                continue
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
        Show the observation that contains local_map, agents_map, planner_map, and neighbors_goal_map
        """

        # _im_local_map = utils.tensor_to_cv(self.local_map, self.map_height, self.map_width)
        # _im_agents_map = utils.tensor_to_cv(self.agents_map, self.map_height, self.map_width)
        # # _im_my_goal_map = utils.tensor_to_cv(self.my_goal_map, self.map_height, self.map_width)
        # _im_planner_map = utils.tensor_to_cv(self.planner_map, self.map_height, self.map_width)
        # _im_neighbors_goal_map = utils.tensor_to_cv(self.neighbors_goal_map, self.map_height, self.map_width)

        # _im_tile = utils.concat_tile_resize([[_im_local_map, _im_agents_map],
        #                                      [_im_planner_map, _im_neighbors_goal_map]],
        #                                      text=[["local_map", "agents_map"],
        #                                            ["planner_map", "neighbors_goal_map"]])
        _im_tile = utils.concat_tile_resize([[self.observation[0].astype('uint8'), self.observation[1].astype('uint8')],
                                             [self.observation[2].astype('uint8'), self.observation[3].astype('uint8')]],
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

        for i in range(0, len(self.init_poses)):
            self.__reset_model_pose("robot{}".format(i), self.init_poses[i][0], self.init_poses[i][1], self.init_poses[i][2])
            self.__reset_model_pose("goal{}".format(i), self.goals[i][0], self.goals[i][1], self.goals[i][2])
        time.sleep(1)
        print("I am robot {}, from ({}, {}, {}) to ({}, {}, {})"\
                .format(self.current_robot_num,\
                        self._current_robot_x, self._current_robot_y, self._current_robot_yaw,\
                        self._current_goal_x, self._current_goal_y, self._current_goal_yaw))

        self.__movebase_client(self.current_robot_num, self._current_goal_x, self._current_goal_y, self._current_goal_yaw)
        time.sleep(1)

        self._current_robot_done = False
        self._done_robots = tuple()
        self._sync_obs_ready = False

        self.init_poses = self.__get_new_poses(self.global_costmap, self.init_poses)
        self.goals = self.__get_new_poses(self.global_costmap, self.goals)
        self._current_goal_x = self.goals[int(self.current_robot_num)][0]    # Agent's goal x
        self._current_goal_y = self.goals[int(self.current_robot_num)][1]    # Agent's goal y
        self._current_goal_yaw = self.goals[int(self.current_robot_num)][2]  # Agent's goal yaw

        return self.observation

    def step(self, u):
        """
        Input: action
        Output: observation, reward, done, info
        """

        done = False
        info = {}
        self._sync_obs_ready = False

        self.__action_to_vel(u)

        # moving cost
        r = -0.3

        # ARRIVED GOAL
        for i, e in enumerate(self.robots_position):
            if utils.dist([e[0], e[1]], [self.goals[i][0], self.goals[i][1]]) <= ARRIVED_RANGE_XY and \
               abs(e[2] - self.goals[i][2]) <= ARRIVED_RANGE_YAW:

                if i == self.current_robot_num:
                    self._current_robot_done = True
                    r = 20

        ## The robot which is in collision will get punishment
        if self._collision:
            self._current_robot_done = True
            r = -2.0
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

    def stop_robot(self):
        """
        Stop current robot
        """
        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        self._pub_vel.publish(msg)

    def __reset_gazebo_world(self):
        """
        Reset gazebo world by call service
        """
        try:
            reset_env = rospy.ServiceProxy('/gazebo/reset_world', EmptySrv)
            reset_env()
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

    def close(self):
        pass

    def __del__(self):
        self.stop_robot()
        # Close cv
        cv2.destroyAllWindows()

    def exit_handler(self, signal_received, frame):
        # Handle any cleanup here
        self.stop_robot()
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        exit(0)

    def __reset_model_pose(self, model_name, x, y, yaw):
        """
        Reset robot's position
        """

        q = quaternion_from_euler(0.0, 0.0, yaw)
        ms = ModelState()
        ms.model_name = model_name
        ms.pose.position.x = x
        ms.pose.position.y = y
        ms.pose.orientation = Quaternion(*q)

        try:
            reset_pose = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            reset_pose(ms)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

    def __movebase_client(self, id, gx, gy, gyaw):
        """
        Call movebase to get path by actionlib
        """

        q = quaternion_from_euler(0.0, 0.0, gyaw)

        client = actionlib.SimpleActionClient('robot{}/move_base'.format(id),MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = gx
        goal.target_pose.pose.position.y = gy
        goal.target_pose.pose.orientation = Quaternion(*q)

        client.send_goal(goal)

    def __get_global_costmap(self, id=0):
        """
        Get global costmap with inflation
        """

        msg = rospy.wait_for_message('/robot{}_move_base/global_costmap/costmap'.format(id), OccupancyGrid)
        map = np.reshape(msg.data, (msg.info.width, msg.info.height)).astype('uint8')
        # cv2.imshow("Global costmap of Agent {}".format(id), map)
        # cv2.waitKey(1)
        return map

    def __get_new_poses(self, map, poses):
        """
        Determine poses of robots and goals for next episode
        Return poses that locate at available position in costmap
        """
        _map = copy.deepcopy(map)
        new_poses = []
        for i in range(0, len(poses)):
            while True:
                _x = int(np.random.rand()*len(_map[0]))
                _y = int(np.random.rand()*len(_map))
                _yaw = np.random.rand()*math.pi
                if _map[_x][_y] == 0: # available
                    __x = _x - len(_map[0])/2
                    __y = (_y - len(_map)/2)*-1
                    new_poses.append((__x*self.map_resolution, __y*self.map_resolution, _yaw))
                    # occupy it
                    x_start = int(_x-self.robot_radius) if int(_x-self.robot_radius)>=0 else 0
                    x_end = int(_x+self.robot_radius) if int(_x+self.robot_radius)<self.map_width else self.map_width
                    y_start = int(_y-self.robot_radius) if int(_y-self.robot_radius)>0 else 0
                    y_end = int(_y+self.robot_radius) if int(_y+self.robot_radius)<self.map_height else self.map_height
                    for ix in range(x_start, x_end):
                        for iy in range(y_start, y_end):
                            _map[ix][iy] = 255
                    break
        return new_poses

