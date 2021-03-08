#!/usr/bin/env bash

TOTAL_ROBOT_NUM=4

screen -X -S turtles_sim quit
for i in $(seq 1 $TOTAL_ROBOT_NUM);
do
  echo "Closeing robot_$i's tf transformer..."
  screen -X -S turtles${i}_tf quit
  echo "Closeing robot_$i's movebase..."
  screen -X -S turtles${i}_movebase quit
done
echo "done. view with screen -ls"
