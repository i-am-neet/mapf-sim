#!/usr/bin/env python3
import sys
import rospy
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry

import torch
from my_a3c import Net, Worker

robot_num = 4

"""
    callback value:
        data[0]: local_costmap
        data[1]: robot0's odom
        data[2]: robot1's odom
        ...
"""
def callback(*data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data[1].pose)
    print(type(data[0].data))

def listener(current_robot_num):

    rospy.init_node('mapf_node', anonymous=True)

    subscribers = []
    sub = message_filters.Subscriber("/robot_"+str(current_robot_num)+"_move_base/local_costmap/costmap", OccupancyGrid)
    print("/robot_"+str(current_robot_num)+"_move_base/local_costmap/costmap")
    subscribers.append(sub)
    for i in range(robot_num):
        sub = message_filters.Subscriber("/robot_"+str(i)+"/odom", Odometry)
        print("/robot_"+str(i)+"/odom")
        subscribers.append(sub)

    ts = message_filters.TimeSynchronizer(subscribers, 10)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        listener(sys.argv[-1])
    else:
        print("Arguments Error...")
