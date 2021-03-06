#!/usr/bin/env bash

screen -X -S turtles_sim quit
for i in {0..3};
do
  echo "Closeing robot_$i's tf transformer..."
  screen -X -S turtles${i}_tf quit
  echo "Closeing robot_$i's movebase..."
  screen -X -S turtles${i}_movebase quit
done
echo "done. view with screen -ls"
