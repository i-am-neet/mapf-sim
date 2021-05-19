#!/usr/bin/env bash

let i=0
while IFS=' ' read -r -a poses_arr;
do
  echo "DELETING model robot$i..."
  screen -dmS rosservice call /gazebo/delete_model "model_name: 'robot$i'"
  sleep 0.5
  ((i++))
done < $(rosls robot_gazebo/config/init_poses.cfg)
