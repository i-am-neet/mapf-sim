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
import utils
import time
import re
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from mapf_pkg.msg import obs, obsArray, float1d, float2d
from mapf_pkg.srv import *

ARRIVED_RANGE_XY = 0.08               # |robots' position - goal's position| < ARRIVED_RANGE_XY (meter)
ARRIVED_RANGE_YAW = math.radians(5)   # |robots' angle - goal's angle| < ARRIVED_RANGE_YAW (degrees to radians)

class MyRosBridge():

    def __init__(self, current_robot_num, robots_num, map_resolution, goals, robot_radius):

        self._current_robot_num = current_robot_num
        self._robots_num = robots_num
        self._map_height = 40
        self._map_width = 40
        self._map_resolution = map_resolution
        self._goals = goals
        self._robot_radius = robot_radius
        self._lidar_range_size = 270
        self._planner_path = None
        self._my_odom = tuple()

        rospy.init_node('mapf_ros_node', anonymous=True)

        # Save costmap
        self.global_costmap = self.__get_global_costmap()

    def obs_converter(self, msg, id):

        _local_costmap = np.asarray(list(msg.observations[id].local_map)).reshape(self._map_height, self._map_width)
        _agents_map = np.asarray(list(msg.observations[id].agents_map)).reshape(self._map_height, self._map_width)
        _planner_map = np.asarray(list(msg.observations[id].planner_map)).reshape(self._map_height, self._map_width)
        _neighbors_goal_map = np.asarray(list(msg.observations[id].neighbors_goal_map)).reshape(self._map_height, self._map_width)

        o = np.stack((_local_costmap, _agents_map, _planner_map, _neighbors_goal_map))

        _observation = dict()
        _observation['map'] = o
        _observation['lidar'] = np.array([msg.observations[id].scan.ranges])
        _rx = msg.observations[id].odom[0]
        _ry = msg.observations[id].odom[1]
        _ryaw = msg.observations[id].odom[2]
        _gx = self.goals[id][0]
        _gy = self.goals[id][1]
        _gyaw = self.goals[id][2]
        _dd = utils.dist([_rx, _ry], [_gx, _gy])
        _dyaw = ((_gyaw - _ryaw) + 2*math.pi) % 2*math.pi
        _observation['goal'] = np.array([[_dd, _dyaw]])
        _observation['plan_len'] = np.array([[len(self._planner_path)]])
        _observation['robot_info'] = np.array([[_rx, _ry, _ryaw]])

        return _observation

    @property
    def get_observation(self):
        """
        Get current robot's observation
        """

        id = self._current_robot_num

        # local costmap
        try:
            _m = rospy.wait_for_message("/observations", obsArray, timeout=5)
        except rospy.ROSException as e:
            print(e)
            return None

        self._my_odom = tuple(_m.observations[id].odom)
        self._planner_path = _m.observations[id].path.poses
        _o = self.obs_converter(_m, id)

        return _o

    @property
    def collision_check(self):
        """
        Get model contact status by gazebo bumper plugin.
        NOTICE: In this data, gets contact pair between models.
        """
        try:
            data = rospy.wait_for_message("/robot{}/bumper".format(str(self._current_robot_num)), ContactsState)
        except rospy.ROSException as e:
            print(e)
            return None
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
        # msg.angular.z = 0
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
        return map

    def get_new_poses(self, refs=None, min_d=None, max_d=None):
        """
        Determine poses of robots and goals for next episode
        Return poses that locate at available position in costmap

        If sets all arguments, get new poses by refs' poses
        min_d < r < max_d, 0 < phi < 3.14159
        new_poses_x = refs_poses_x + r*cos(phi)
        new_poses_y = refs_poses_y + r*sin(phi)
        """
        USE_REFS = False
        if any([refs, min_d, max_d]) & all([refs, min_d, max_d]):
            USE_REFS = True
        elif any([refs, min_d, max_d]):
            raise ValueError("three arguments required!")

        _map = copy.deepcopy(self.global_costmap)
        new_poses = []
        for i in range(0, self._robots_num):
            while True:
                if USE_REFS:
                    # create new poses by image coordinate
                    _r = np.random.uniform(low=min_d, high=max_d)
                    _phi = np.random.uniform(low=0, high=2*np.pi)
                    _x = int((refs[i][0] + _r*np.cos(_phi))/self._map_resolution + len(_map[0])/2)
                    _y = int(len(_map)/2 - (refs[i][1] + _r*np.sin(_phi))/self._map_resolution)
                    if _x < 0 or _x >= len(_map[0]) or _y < 0 or _y >= len(_map):
                        continue
                else:
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
        # _odom = rospy.wait_for_message("/robot{}/mobile/odom".format(str(self._current_robot_num)), Odometry)
        # _rx = _odom.pose.pose.position.x
        # _ry = _odom.pose.pose.position.y
        # _px = self._planner_path[-1].pose.position.x
        # _py = self._planner_path[-1].pose.position.y
        # _x = _px - _rx
        # _y = _py - _ry
        # v = np.array([_x, _y], dtype=np.float32)
        # v[v>1] = 1
        # v[v<-1] = -1
        # return v

    def reset_poses(self, inits, goals):
        # Reset poses on Gazebo ONLY
        for i in range(0, self._robots_num):
            self.__reset_model_pose("robot{}".format(i), inits[i][0], inits[i][1], inits[i][2])
            self.__reset_model_pose("goal{}".format(i), goals[i][0], goals[i][1], goals[i][2])
            # self.__reset_model_pose("robot{}".format(i), inits[i][0], inits[i][1], 0)
            # self.__reset_model_pose("goal{}".format(i), goals[i][0], goals[i][1], 0)

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
        self.change_goals_client(new_goals)
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

    def change_goals_client(self, ng):
        rospy.wait_for_service('change_goals')
        mm = float2d()
        for i, e in enumerate(ng):
            m = float1d(e)
            mm.data.append(m)
        try:
            add_two_ints = rospy.ServiceProxy('change_goals', ChangeGoals)
            resp1 = add_two_ints(mm)
            return resp1.success
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
