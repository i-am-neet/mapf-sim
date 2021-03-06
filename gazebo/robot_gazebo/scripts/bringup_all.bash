#!/usr/bin/env bash

let i=0
while IFS=' ' read -r -a poses_arr;
do
  echo "SPAWNING robot$i, pose at: (${poses_arr[0]}, ${poses_arr[1]}, ${poses_arr[2]})..."
  screen -dmS spawn_robot${i} roslaunch robot_gazebo spawn_robot.launch id:=$i \
                              x_pos:=${poses_arr[0]} y_pos:=${poses_arr[1]} yaw_pos:=${poses_arr[2]}
  sleep 0.5
  echo "LAUNCHING robot$i's tf transformer 'robot${i}_tf'..."
  # ros-noetic
  screen -dmS robot${i}_tf roslaunch robot_gazebo tf2_map_odom.launch ns:=robot${i}_tf odom_frame:=robot${i}/odom
  sleep 0.5
  echo "LAUNCHING robot$i's movebase 'robot${i}_movebase'..."
  screen -dmS robot${i}_movebase roslaunch robot_gazebo movebase.launch robot:=robot$i \
                                 base_frame:=base_link odom_frame:=odom
  sleep 0.5
  ((i++))
done < $(rosls robot_gazebo/config/init_poses.cfg)
