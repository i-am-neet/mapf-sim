import math
import copy
import numpy as np

def draw_robot(map, map_width, map_height, rx, ry, robot_radius, map_resolution):
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

def draw_goal(map, map_width, map_height, gx, gy, radius, map_resolution):
    return draw_robot(map, map_width, map_height, gx, gy, radius, map_resolution)

def draw_neighbors_goal(map, map_width, map_height, ngx, ngy, myx, myy, robot_radius, map_resolution):
    _r = robot_radius/map_resolution
    dx = ngx - myx
    dy = ngy - myy
    m = dy/dx if dx != 0 else math.inf
    if abs(dx) <= map_width/2-_r and abs(dy) <= map_height/2-_r:     # in the FOV
        return draw_robot(map, map_width, map_height, dx, dy, robot_radius, map_resolution)
    elif m == math.inf:                                              # goal on vertical
        vy = np.sign(dy)*map_height/2
        return draw_robot(map, map_width, map_height, 0, vy, robot_radius, map_resolution)
    elif abs(m) <= 1:                                                # slope(m) < 1
        vx = np.sign(dx)*map_width/2                                 # draw on x-axis edge of fov
        vy = m*vx                                                    # y=mx+b. bias(b) is 0, because (0, 0) is self
        return draw_robot(map, map_width, map_height, vx, vy, robot_radius, map_resolution)
    elif abs(m) > 1:                                                 # slope(m) < 1
        vy = np.sign(dy)*map_height/2                                # draw on y-axis edge of fov
        vx = vy/m
        return draw_robot(map, map_width, map_height, vx, vy, robot_radius, map_resolution)

def draw_path(map, map_width, map_height, myx, myy, path, map_resolution):

    _map = copy.deepcopy(map)

    for p in path:
        # get relative pose according to robot's position
        px = p.pose.position.x / map_resolution
        py = p.pose.position.y / map_resolution

        dx = px - myx
        dy = py - myy

        dx = int(dx + map_width/2)
        dy = int(map_height/2 - dy)

        if dx >= 0 and dx < map_width and dy >= 0 and dy < map_height:     # in the FOV
            _map[int(dx+map_width*dy)] = 255
        else: # out of FOV
            break

    return _map
