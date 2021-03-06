#!/usr/bin/env bash

for i in {0..3};
do
  echo "Closeing robot_$i's tf transformer..."
  screen -X -S turtles${i}_tf quit
done
echo "done. view with screen -ls"
