#!/usr/bin/env bash

echo "LAUNCHING turtles..."
screen -dmS turtles_sim roslaunch robot_stage turtles.launch
sleep 3
for i in {0..3};
do
  echo "LAUNCHING robot_$i's tf transformer..."
  screen -dmS turtles${i}_tf roslaunch robot_stage tf2_broadcaster.launch ns:=$i robot:=robot_$i
done
echo "done. view with screen -ls"