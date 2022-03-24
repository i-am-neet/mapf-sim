#!/usr/bin/env python3
import math
import numpy as np
import rospy
import rosnode
import message_filters
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from mapf_pkg.msg import obs, obsArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import utils
from matplotlib import pyplot as plt
import time
import re
import cv2
import os
from signal import signal, SIGINT
from mapf_ros_node import MyRosBridge
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import actionlib

def movebase_client(id, gx, gy, gyaw):
    """
    Call movebase to get path by actionlib
    """

    q = quaternion_from_euler(0.0, 0.0, gyaw)

    client = actionlib.SimpleActionClient('robot{}/move_base'.format(id), MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = gx
    goal.target_pose.pose.position.y = gy
    goal.target_pose.pose.orientation = Quaternion(*q)

    client.send_goal(goal)

_pub_obs = rospy.Publisher('/observations', obsArray, queue_size=1)

robots_num = rospy.get_param("/robots_num")
robot_radius = rospy.get_param("/robot_radius")
goals = utils.read_file(rospy.get_param("/goals_init_file"))
map_resolution = rospy.get_param("/map_resolution")
map_width = 40
map_height = 40

_lasers = []
_odoms = []
_local_maps = []
_planners = []
_o = obs()
_o_arr = obsArray()

def my_observations():

    _o_arr.observations.clear()
    _lasers.clear()
    _odoms.clear()
    _local_maps.clear()
    _planners.clear()

    for i in range(0, robots_num):
        try:
            _m = rospy.wait_for_message("/robot{}/laser".format(str(i)), LaserScan, timeout=5)
            _lasers.append(_m)
        except rospy.ROSException as e:
            print(e)
            return None
        try:
            _m = rospy.wait_for_message("/robot{}/mobile/odom".format(str(i)), Odometry, timeout=5)
            _x = _m.pose.pose.position.x
            _y = _m.pose.pose.position.y
            _yaw = euler_from_quaternion([_m.pose.pose.orientation.x,
                                            _m.pose.pose.orientation.y,
                                            _m.pose.pose.orientation.z,
                                            _m.pose.pose.orientation.w])[2]
            _odoms.append((_x, _y, _yaw))
        except rospy.ROSException as e:
            print(e)
            return None
        try:
            _m = rospy.wait_for_message("/robot{}_move_base/local_costmap/costmap".format(str(i)), OccupancyGrid, timeout=5)
            # _l_m = np.asarray(_m.data)
            _l_m = np.array(_m.data, dtype='uint8')
            _l_m[_l_m < 10] = 0
            _l_m[_l_m >= 10] = 255
            _l_m = _l_m.reshape(map_height, map_width)[::-1].reshape(-1)
            _local_maps.append(_l_m)
        except rospy.ROSException as e:
            print(e)
            return None
        try:
            _m = rospy.wait_for_message("/robot{}_move_base/NavfnROS/plan".format(str(i)), Path, timeout=5)
            _planners.append(_m)
        except rospy.ROSException as e:
            print(e)
            return None

    for i in range(0, robots_num):

        _o.scan = _lasers[i]
        _o.path = _planners[i]
        _o.local_map = _local_maps[i].tolist()

        _x = _odoms[i][0] / map_resolution
        _y = _odoms[i][1] / map_resolution
        _yaw = _odoms[i][2]

        _planner_map = np.zeros(_local_maps[i].size)
        _agents_map = np.zeros(_local_maps[i].size)
        _neighbors_goal_map = np.zeros(_local_maps[i].size)
        _planner_map = utils.draw_path(_planner_map, map_width, map_height, _x, _y, _planners[i].poses, map_resolution)
        _o.planner_map = _planner_map.astype('uint8').tolist()


        # # Scale unit for pixel with map's resolution (meter -> pixels)
        my_x = _odoms[i][0] / map_resolution
        my_y = _odoms[i][1] / map_resolution
        my_yaw = _odoms[i][2]

        for j, e in enumerate(_odoms):
            _rx = e[0] / map_resolution
            _ry = e[1] / map_resolution

            if abs(_rx - my_x) <= map_width/2 and abs(_ry - my_y) <= map_height/2:

                _agents_map = utils.draw_robot(_agents_map, map_width, map_height, _rx - my_x, _ry - my_y, robot_radius, map_resolution)
                _o.agents_map = _agents_map.astype('uint8').tolist()

                # Neighbors
                if i != int(j):
                    print(i)
                    print(j)
                    print(len(goals))
                    _ngx = goals[j][0] / map_resolution     # Neighbor's goal x
                    _ngy = goals[j][1] / map_resolution     # Neighbor's goal y
                    _ngyaw = goals[j][2]                         # Neighbor's goal yaw
                    _neighbors_goal_map = utils.draw_neighbors_goal(_neighbors_goal_map, map_width, map_height, _ngx, _ngy, my_x, my_y, robot_radius, map_resolution)
                    _o.neighbors_goal_map = _neighbors_goal_map.astype('uint8').tolist()

        _o_arr.observations.append(_o)

    _pub_obs.publish(_o_arr)

if __name__ == "__main__":

    rospy.init_node('mapf_obs_node', anonymous=True)
    r = rospy.Rate(100)

    print("Sending move_base's goal:")
    for i in range(0, robots_num):
        print("robot {} will go to ({}, {}, {})".format(i, goals[i][0], goals[i][1], goals[i][2]))
        movebase_client(i, goals[i][0], goals[i][1], goals[i][2])

    print("Start getting observations")
    while not rospy.is_shutdown():
        my_observations()
        r.sleep()
