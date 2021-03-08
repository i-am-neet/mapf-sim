#!/usr/bin/env bash

TOTAL_ROBOT_NUM=4

echo "LAUNCHING turtles..."
screen -dmS turtles_sim roslaunch robot_stage turtles.launch
sleep 3
for i in $(seq 1 $TOTAL_ROBOT_NUM);
do
  echo "LAUNCHING robot_$i's tf transformer..."
  screen -dmS turtles${i}_tf roslaunch robot_stage tf2_broadcaster.launch ns:=$i robot:=R$i
done
sleep 3
for i in $(seq 1 $TOTAL_ROBOT_NUM);
do
  echo "LAUNCHING robot_$i's movebase..."
  # screen -dmS turtles${i}_movebase roslaunch robot_stage turtle_movebase.launch ns:=$i robot:=R1
  screen -dmS turtles${i}_movebase roslaunch robot_stage turtle_movebase.launch robot:=R$i
  # roslaunch robot_stage turtle_movebase.launch ns:=stage/R1 robot:=R1
done
echo "done. view with screen -ls"
