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

class MyObservation():

    def __init__(self, robots_num,
                       robot_radius,
                       goals,
                       map_resolution):

        if len(goals) != robots_num:
            raise ValueError("The amount of goals '%d' must equal to robots_num '%d" %(len(goals), robots_num))

        signal(SIGINT, self.exit_handler)

        # Initialize variables of Environment
        self.goals = goals
        self.robots_num = robots_num
        self.robot_radius = robot_radius
        self.map_resolution = map_resolution

        rospy.init_node('mapf_obs_node', anonymous=True)

        # Initialize
        for i in range(0, self.robots_num):
            movebase_client(i, self.goals[i][0], self.goals[i][1], self.goals[i][2])

        # Publisher
        self._pub_obs = rospy.Publisher('/observations', obsArray, queue_size=1)

        # Subscriber
        _subscribers = []
        print("Laser scans")
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}/laser".format(str(i)), LaserScan)
            print("\t/robot{}/laser".format(str(i)))
            _subscribers.append(_sub_obs)
        print("Odoms")
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}/mobile/odom".format(str(i)), Odometry)
            print("\t/robot{}/mobile/odom".format(str(i)))
            _subscribers.append(_sub_obs)
        print("Local costmaps")
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}_move_base/local_costmap/costmap".format(str(i)), OccupancyGrid)
            print("\t/robot{}_move_base/local_costmap/costmap".format(str(i)))
            _subscribers.append(_sub_obs)
        print("Planner paths")
        for i in range(0, self.robots_num):
            _sub_obs = message_filters.Subscriber("/robot{}_move_base/NavfnROS/plan".format(str(i)), Path)
            print("\t/robot{}_move_base/NavfnROS/plan".format(str(i)))
            _subscribers.append(_sub_obs)

        # ts = message_filters.TimeSynchronizer(_subscribers, 10)
        ts = message_filters.ApproximateTimeSynchronizer(_subscribers, 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.__callback)

        rospy.spin()

    def __callback(self, *data):
        """
        N is amount of robots
        callback value:
            data[0]: robot1's laser_scan
            data[N-1]: robotN's laser_scan
            data[N]: robot1's odom
            data[2N-1]: robotN's odom
            data[2N]: robot1's local_map
            data[3N-1]: robotN's local_map
            data[3N]: robot1's planner path
            data[4N-1]: robotN's planner path

        Observations are created by this function
        mapf_pkg/obs.msg
            sensor_msgs/LaserScan scan
            nav_msgs/Path path
            nav_msgs/OccupancyGrid local_map
            nav_msgs/OccupancyGrid agents_map
            nav_msgs/OccupancyGrid neighbors_goal_map
            nav_msgs/OccupancyGrid planner_map

        mapf_pkg/obsArry.msg
            obs[] observations
        """
        _o = obs()
        _o_arr = obsArray()

        for i in range(0, self.robots_num):
            # scan
            _o.scan = data[i]
            # path
            _o.path = data[3*self.robots_num + i]
            # local_map
            _l_m = np.asarray(data[2*self.robots_num + i].data) # type(_l_m) is ndarray
            _l_m[_l_m < 10] = 0
            _l_m[_l_m >= 10] = 255
            # _o.local_map = _l_m.reshape(40, 40)[::-1].reshape(-1).tolist()
            _o.local_map = [1,2,3]

            _current_robot_x = data[self.robots_num + i].pose.pose.position.x
            _current_robot_y = data[self.robots_num + i].pose.pose.position.y
            _current_robot_yaw = euler_from_quaternion([data[self.robots_num + i].pose.pose.orientation.x,
                                                        data[self.robots_num + i].pose.pose.orientation.y,
                                                        data[self.robots_num + i].pose.pose.orientation.z,
                                                        data[self.robots_num + i].pose.pose.orientation.w])[2]
            # _o.agents_map
            # _o.neighbors_goal_map
            # _o.planner_map

            _o_arr.observations.append(_o)

        self._pub_obs.publish(_o_arr)

    def close(self):
        pass

    def __del__(self):
        cv2.destroyAllWindows()
        pass

    def exit_handler(self, signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        exit(0)

if __name__ == "__main__":
    robots_num = rospy.get_param("/robots_num")
    robot_radius = rospy.get_param("/robot_radius")
    goals = utils.read_file(rospy.get_param("/goals_init_file"))
    print(goals)
    map_resolution = rospy.get_param("/map_resolution")
    MyObservation(robots_num, robot_radius, goals, map_resolution)
