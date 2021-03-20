#!/usr/bin/env python3
import sys
from signal import signal, SIGINT
from sys import exit
import math
import copy
import numpy as np
import rospy
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry
import cv2

import torch
from my_a3c import Net, Worker

current_robot_num = 0
robots_num = 4
robot_diameter = 0.25 # according size of tb3: 0.3*0.3*0.4
robot_radius = robot_diameter/2
map_resolution = 0.01

## Tmp
goals = [(2, 2), (-1.5, 1.5), (-1.5, -1.5), (2, -2)]

def draw_robot(map, map_width, map_height, rx, ry, stick_on_bound=False):
    _r = robot_radius/map_resolution
    rx = rx + map_width/2
    ry = map_height/2 - ry

    x_start = int(rx-_r) if int(rx-_r)>=0 else 0
    x_end = int(rx+_r) if int(rx+_r)<map_width else map_width
    y_start = int(ry-_r) if int(ry-_r)>0 else 0
    y_end = int(ry+_r) if int(ry+_r)<map_height else map_height

    _map = copy.deepcopy(map)

    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            _map[int(x+map_width*y)] = 255

    return _map

def draw_goal(map, map_width, map_height, gx, gy):
    return draw_robot(map, map_width, map_height, gx, gy)

def draw_neighbors_goal(map, map_width, map_height, ngx, ngy, myx, myy):
    _r = robot_radius/map_resolution
    dx = ngx - myx
    dy = ngy - myy
    m = dy/dx if dx != 0 else math.inf
    if abs(dx) <= map_width/2-_r and abs(dy) <= map_height/2-_r:     # in the FOV
        return draw_robot(map, map_width, map_height, dx, dy, True)
    elif m == math.inf:                                              # goal on vertical
        vy = np.sign(dy)*map_height/2
        return draw_robot(map, map_width, map_height, 0, vy, True)
    elif abs(m) <= 1:                                                # slope(m) < 1
        vx = np.sign(dx)*map_width/2                                 # draw on x-axis edge of fov
        vy = m*vx                                                    # y=mx+b. bias(b) is 0, because (0, 0) is self
        return draw_robot(map, map_width, map_height, vx, vy, True)
    elif abs(m) > 1:                                                 # slope(m) < 1
        vy = np.sign(dy)*map_height/2                                # draw on y-axis edge of fov
        vx = vy/m
        return draw_robot(map, map_width, map_height, vx, vy, True)

"""
    callback value:
        data[0]: local_costmap
        data[1]: robot1's odom
        data[2]: robot2's odom
        ...
"""
def callback(*data):
    ## local costmap: tuple -> np.array -> tensor
    map_width = data[0].info.width
    map_height = data[0].info.height
    local_map = torch.from_numpy(np.asarray(data[0].data).reshape(map_height, map_width)[::-1].reshape(-1))

    ## Current robot's info
    my_x = data[int(current_robot_num)].pose.pose.position.x / map_resolution
    my_y = data[int(current_robot_num)].pose.pose.position.y / map_resolution

    ## Initial size equal to local costmap
    my_goal_map = torch.zeros(local_map.size())
    agents_map = torch.zeros(local_map.size())
    neighbors_goal_map = torch.zeros(local_map.size())

    agx = goals[int(current_robot_num)-1][0] / map_resolution
    agy = goals[int(current_robot_num)-1][1] / map_resolution

    my_goal_map = draw_goal(my_goal_map, map_width, map_height, agx - my_x, agy - my_y)

    for i, e in enumerate(data):
        if i != 0: # data[0] is local costmap
            _rx = e.pose.pose.position.x / map_resolution
            _ry = e.pose.pose.position.y / map_resolution
            if abs(_rx - my_x) <= map_width/2 and abs(_ry - my_y) <= map_height/2:

                agents_map = draw_robot(agents_map, map_width, map_height, _rx - my_x, _ry - my_y)

                ## Neighbors
                if i != int(current_robot_num):
                    _ngx = goals[i-1][0] / map_resolution
                    _ngy = goals[i-1][1] / map_resolution
                    neighbors_goal_map = draw_neighbors_goal(neighbors_goal_map, map_width, map_height, _ngx, _ngy, my_x, my_y)

    cv2.imshow("local_map", local_map.reshape(map_height, map_width).cpu().numpy().astype('uint8'))
    cv2.imshow("agents_map", agents_map.reshape(map_height, map_width).cpu().numpy().astype('uint8'))
    cv2.imshow("my_goal_map", my_goal_map.reshape(map_height, map_width).cpu().numpy().astype('uint8'))
    cv2.imshow("neighbors_goal_map", neighbors_goal_map.reshape(map_height, map_width).cpu().numpy().astype('uint8'))
    cv2.waitKey(1)

def listener(current_robot_num):

    rospy.init_node('mapf_node', anonymous=True)

    subscribers = []
    sub = message_filters.Subscriber("/R"+str(current_robot_num)+"_move_base/local_costmap/costmap", OccupancyGrid)
    print("/R"+str(current_robot_num)+"_move_base/local_costmap/costmap")
    subscribers.append(sub)
    for i in range(1, robots_num+1):
        sub = message_filters.Subscriber("/stage/R"+str(i)+"/odometry", Odometry)
        print("/stage/R"+str(i)+"/odometry")
        subscribers.append(sub)

    ts = message_filters.TimeSynchronizer(subscribers, 10)
    ts.registerCallback(callback)

    rospy.spin()

def exit_handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':

    signal(SIGINT, exit_handler)

    if len(sys.argv) == 2:
        current_robot_num = sys.argv[-1]
        listener(current_robot_num)
    else:
        print("Arguments Error...")
