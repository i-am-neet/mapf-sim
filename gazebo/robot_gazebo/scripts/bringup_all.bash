#!/usr/bin/env bash

# TOTAL_ROBOT_NUM=3

#echo "LAUNCHING turtles..."
#screen -dmS turtles_sim roslaunch robot_stage turtles.launch
#sleep 3
# for i in $(seq 0 $TOTAL_ROBOT_NUM);
# do
#   echo "LAUNCHING robot_$i's tf transformer..."
#   screen -dmS turtles${i}_tf roslaunch robot_stage tf2_broadcaster.launch ns:=$i robot:=robot_$i
# done
# sleep 3
# for i in $(seq 0 $TOTAL_ROBOT_NUM);
# do
#   echo "LAUNCHING robot_$i's movebase..."
#   # screen -dmS turtles${i}_movebase roslaunch robot_stage turtle_movebase.launch ns:=$i robot:=R1
#   screen -dmS turtles${i}_movebase roslaunch robot_stage turtle_movebase.launch robot:=robot_$i
#   # roslaunch robot_stage turtle_movebase.launch ns:=stage/R1 robot:=R1
# done
# echo "done. view with screen -ls"

# for i in $(seq 0 $TOTAL_ROBOT_NUM);
# do
#   echo "SPAWNING robot$i..."
#   # screen -dmS turtles${i}_movebase roslaunch robot_stage turtle_movebase.launch ns:=$i robot:=R1
#   screen -dmS spawn_robot${i} roslaunch robot_gazebo spawn-robot.launch id:=$i
#   # roslaunch robot_stage turtle_movebase.launch ns:=stage/R1 robot:=R1
# done
# echo "done. view with screen -ls"

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
