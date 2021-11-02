#!/usr/bin/env python3
import copy
import math
import numpy as np
import rospy
import rosnode
import message_filters
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_srvs.srv import Empty as EmptySrv
from geometry_msgs.msg import Twist, Pose, Quaternion
from std_msgs.msg import Bool
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv2
import utils
import time
import re
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY (meter)
ARRIVED_RANGE_YAW = math.radians(5)   # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW (degrees to radians)

MAX_SPEED = abs(0.5) # maximum speed (m/s)

class MyRosBridge():

    def __init__(self, current_robot_num, robots_num, map_resolution, goals, robot_radius):

        self._current_robot_num = current_robot_num
        self._robots_num = robots_num
        self._map_height = 0
        self._map_width = 0
        self._map_resolution = map_resolution
        self._goals = goals
        self._robot_radius = robot_radius
        self._lidar_range_size = 0
        self._planner_path = None
        self._my_odom = tuple()

        rospy.init_node('mapf_ros_node', anonymous=True)

        # # Subscriber
        # # Register all topics for message_filters
        # _subscribers = []
        # # local costmap
        # for i in range(0, self._robots_num):
        #     _sub_obs = message_filters.Subscriber("/robot{}_move_base/local_costmap/costmap".format(str(i)), OccupancyGrid)
        #     print("/robot{}_move_base/local_costmap/costmap".format(str(i)))
        #     _subscribers.append(_sub_obs)
        # # global planner
        # for i in range(0, self._robots_num):
        #     _sub_obs = message_filters.Subscriber("/robot{}_move_base/NavfnROS/plan".format(str(i)), Path)
        #     print("/robot{}_move_base/NavfnROS/plan".format(str(i)))
        #     _subscribers.append(_sub_obs)
        # # lidar
        # for i in range(0, self._robots_num):
        #     _sub_obs = message_filters.Subscriber("/robot{}/laser".format(str(i)), LaserScan)
        #     print("/robot{}/laser".format(str(i)))
        #     _subscribers.append(_sub_obs)
        # # odom
        # for i in range(0, self._robots_num):
        #     _sub_obs = message_filters.Subscriber("/robot{}/mobile/odom".format(str(i)), Odometry)
        #     print("/robot{}/mobile/odom".format(str(i)))
        #     _subscribers.append(_sub_obs)

        # ts = message_filters.ApproximateTimeSynchronizer(_subscribers, 10, 0.1, allow_headerless=True)
        # ts.registerCallback(self.__callback)

        # while self._map_height == 0:
        #     print("waiting ROS message_filters...")
        #     time.sleep(1)

        # Waiting until all mapf_env_node started
        rosnode.rosnode_cleanup()
        _check_regex = re.compile("/mapf_ros_node*")
        MAPF_ALIVE_NODES = list()
        while len(MAPF_ALIVE_NODES) != self._robots_num:
            print("waiting all mapf_ros_node...")
            rosnode.rosnode_cleanup()
            MAPF_ALIVE_NODES.clear()
            MAPF_NODES = list(filter(_check_regex.match, rosnode.get_node_names()))
            for n in MAPF_NODES:
                if rosnode.rosnode_ping(n, max_count=3):
                    MAPF_ALIVE_NODES.append(n)
        time.sleep(1)

        # Save costmap
        self.global_costmap = self.__get_global_costmap()

    @property
    def get_observation(self):
        """
        Get current robot's observation
        """

        id = self._current_robot_num

        # local costmap
        _m = rospy.wait_for_message("/robot{}_move_base/local_costmap/costmap".format(str(id)), OccupancyGrid)
        self._map_width = _m.info.width
        self._map_height = _m.info.height
        _l_m = np.asarray(_m.data)
        _l_m[_l_m < 10] = 0
        _l_m[_l_m >= 10] = 255
        _local_costmap = _l_m.reshape(self._map_height, self._map_width)[::-1].reshape(-1)
        # global planner
        _global_planner = rospy.wait_for_message("/robot{}_move_base/NavfnROS/plan".format(str(id)), Path)
        self._planner_path = _global_planner.poses
        # lidar
        _lidar_data = rospy.wait_for_message("/robot{}/laser".format(str(id)), LaserScan)
        # odom
        _robots_odom = []
        for i in range(0, self._robots_num):
            _odom = rospy.wait_for_message("/robot{}/mobile/odom".format(str(i)), Odometry)
            _x = _odom.pose.pose.position.x
            _y = _odom.pose.pose.position.y
            _yaw = euler_from_quaternion([_odom.pose.pose.orientation.x,
                                          _odom.pose.pose.orientation.y,
                                          _odom.pose.pose.orientation.z,
                                          _odom.pose.pose.orientation.w])[2]
            _robots_odom.append((_x, _y, _yaw))

        # Scale unit for pixel with map's resolution (meter -> pixels)
        my_x = _robots_odom[id][0] / self._map_resolution
        my_y = _robots_odom[id][1] / self._map_resolution
        my_yaw = _robots_odom[id][2]

        self._my_odom = (_robots_odom[id][0], _robots_odom[id][1], _robots_odom[id][2])

        # Initialize size equal to local costmap
        _planner_map = np.zeros(_local_costmap.size)
        _agents_map = np.zeros(_local_costmap.size)
        _neighbors_goal_map = np.zeros(_local_costmap.size)

        _planner_map = utils.draw_path(_planner_map, self._map_width, self._map_height, my_x, my_y, self._planner_path, self._map_resolution)

        for i, e in enumerate(_robots_odom):
            _rx = e[0] / self._map_resolution
            _ry = e[1] / self._map_resolution

            if abs(_rx - my_x) <= self._map_width/2 and abs(_ry - my_y) <= self._map_height/2:

                _agents_map = utils.draw_robot(_agents_map, self._map_width, self._map_height, _rx - my_x, _ry - my_y, self._robot_radius, self._map_resolution)

                # Neighbors
                if i != int(id):
                    _ngx = self._goals[i][0] / self._map_resolution     # Neighbor's goal x
                    _ngy = self._goals[i][1] / self._map_resolution     # Neighbor's goal y
                    _ngyaw = self._goals[i][2]                         # Neighbor's goal yaw
                    _neighbors_goal_map = utils.draw_neighbors_goal(_neighbors_goal_map, self._map_width, self._map_height, _ngx, _ngy, my_x, my_y, self._robot_radius, self._map_resolution)

        # Reshape map to 2-dims
        _local_costmap = _local_costmap.reshape(self._map_height, self._map_width)
        _agents_map = _agents_map.reshape(self._map_height, self._map_width)
        _planner_map = _planner_map.reshape(self._map_height, self._map_width)
        _neighbors_goal_map = _neighbors_goal_map.reshape(self._map_height, self._map_width)

        # Observation is stack all map tensor, then convert to np.array
        o = np.stack((_local_costmap, _agents_map, _planner_map, _neighbors_goal_map))

        _observation = dict()
        _observation['map'] = o
        _observation['lidar'] = [_lidar_data]
        _observation['goal'] = [[self._goals[id][0] - _robots_odom[id][0],
                                 self._goals[id][1] - _robots_odom[id][1],
                                 self._goals[id][2] - _robots_odom[id][2]]]

        return _observation

    @property
    def collision_check(self):
        """
        Get model contact status by gazebo bumper plugin.
        NOTICE: In this data, gets contact pair between models.
        """
        data = rospy.wait_for_message("/robot{}/bumper".format(str(self._current_robot_num)), ContactsState)
        _collision = False
        # print("Collision Length: {}".format(len(data.states)))
        for i, e in enumerate(data.states):
            # print("Pair {}: {} <---> {}".format(i, e.collision1_name, e.collision2_name))
            A = [e.collision1_name, e.collision2_name]

            if any('ground_plane' in a.lower() for a in A):
                # Ignore collision with ground_plane
                continue
            elif any('wall' in a.lower() for a in A):
                # print("{} Hit the wall!!!!!".format(c))
                _collision = True
                break
            elif any('door' in a.lower() for a in A):
                # print("{} Hit the door!!!!!".format(c))
                _collision = True
                break
            elif all('robot' in a.lower() for a in A):
                # print("{} Hit other robot!!".format(c))
                _collision = True
                break
            else:
                raise Exception("Unknown collision condition, collision pair:\n {} <---> {}".format(A[0], A[1]))

        return _collision

    def action_to_vel(self, action):
        msg = Twist()
        msg.linear.x = action[0]
        msg.linear.y = action[1]
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = action[2]
        _pub_vel = rospy.Publisher('/robot{}/mobile/cmd_vel'.format(self._current_robot_num), Twist, queue_size=1)
        _pub_vel.publish(msg)

    def stop_robots(self):
        """
        Stop all robots
        """
        msg = Twist()
        msg.linear.x = 0
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        for i in range(0, self._robots_num):
            _pub_vel = rospy.Publisher('/robot{}/mobile/cmd_vel'.format(i), Twist, queue_size=1)
            _pub_vel.publish(msg)

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

    def movebase_client(self, gx, gy, gyaw):
        """
        Call movebase to get path by actionlib
        """

        q = quaternion_from_euler(0.0, 0.0, gyaw)

        client = actionlib.SimpleActionClient('robot{}/move_base'.format(self._current_robot_num),MoveBaseAction)
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

        print("Get Global Costmap")
        msg = rospy.wait_for_message('/robot{}_move_base/global_costmap/costmap'.format(id), OccupancyGrid)
        map = np.reshape(msg.data, (msg.info.width, msg.info.height)).astype('uint8')
        # cv2.imshow("Global costmap of Agent {}".format(id), map)
        # cv2.waitKey(1)
        return map

    def get_new_poses(self):
        """
        Determine poses of robots and goals for next episode
        Return poses that locate at available position in costmap
        """
        _map = copy.deepcopy(self.global_costmap)
        new_poses = []
        for i in range(0, self._robots_num):
            while True:
                _x = int(np.random.rand()*len(_map[0]))
                _y = int(np.random.rand()*len(_map))
                _yaw = np.random.rand()*math.pi
                if _map[_x][_y] == 0: # available
                    __x = _x - len(_map[0])/2
                    __y = (_y - len(_map)/2)*-1
                    new_poses.append((__x*self._map_resolution, __y*self._map_resolution, _yaw))
                    # occupy it
                    x_start = int(_x-self._robot_radius) if int(_x-self._robot_radius)>=0 else 0
                    x_end = int(_x+self._robot_radius) if int(_x+self._robot_radius)<self._map_width else self._map_width
                    y_start = int(_y-self._robot_radius) if int(_y-self._robot_radius)>0 else 0
                    y_end = int(_y+self._robot_radius) if int(_y+self._robot_radius)<self._map_height else self._map_height
                    for ix in range(x_start, x_end):
                        for iy in range(y_start, y_end):
                            _map[ix][iy] = 255
                    break
        return new_poses

    @property
    def check_movebase(self):
        msg = rospy.wait_for_message('/robot{}/move_base/status'.format(self._current_robot_num), GoalStatusArray)
        for m in msg.status_list:
            if m.status == 4:
                print("Movebase Aborted... Restarting")
                return False
        return True

    @property
    def expert_action(self):
        msg = rospy.wait_for_message('/robot{}/move_base/cmd_vel'.format(self._current_robot_num), Twist)
        return np.array([msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def reset_poses(self, inits, goals):
        # Reset poses
        for i in range(0, self._robots_num):
            self.__reset_model_pose("robot{}".format(i), inits[i][0], inits[i][1], inits[i][2])
            self.__reset_model_pose("goal{}".format(i), goals[i][0], goals[i][1], goals[i][2])

    def clear_map(self):

        for i in range(0, self._robots_num):
            # Clear costmap
            try:
                clear_map = rospy.ServiceProxy('/robot{}_move_base/clear_costmaps'.format(i), EmptySrv)
                clear_map()
            except rospy.ServiceException as e:
                print("Service call failed: {}".format(e))

    @property
    def goals(self):
        return self._goals

    @goals.setter
    def goals(self, new_goals):
        self._goals = new_goals

    @property
    def map_height(self):
        return self._map_height

    @property
    def map_width(self):
        return self._map_width

    @property
    def lidar_range_size(self):
        return self._lidar_range_size

    @property
    def planner_path(self):
        return self._planner_path

    @property
    def odom(self):
        return self._my_odom
