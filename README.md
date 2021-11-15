# System
Name                        | Version      
----------------------------|-------:
Ubuntu                      | 20.04
Robot Operating System (ROS)| noetic
Python                      | 3.8
Pip                         | 20.0.2
Nvidia driver               | 460.91.03
CUDA                        | 10.1
cUDNN                       | 7.6
Gazebo                      | 11.5.1

# Setup
## Create python environment
```bash
$ virtualenv -p python3.6 $HOME/mapf
$ source $HOME/mapf/bin/active
```
## Install requirement package
Simulator:
```bash
$ sudo apt install screen python3-catkin-pkg
$ sudo apt install ros-noetic-hector-sensors-description
$ sudo apt install ros-noetic-map-server
$ sudo apt install ros-noetic-move-base
```

Pytorch:
```bash
$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Python:
```bash
$ pip install -r requirements.txt
```

## Clone & Build this project
```bash
$ cd $HOME
$ mkdir -p mapf_ws/src && cd mapf_ws
$ git clone --recursive https://github.com/i-am-neet/mapf-sim.git src/
$ catkin_make
```

# Startup
```bash
$ cd $HOME/mapf_ws && source devel/setup.bash
# Start RL env of simualtor
$ roslaunch robot_gazebo home_area.launch gui:=false # or empty.launch
# Bringup robots
# There are 2 arguments for bringup_all.bash:
# $1:PATH of INIT_POSES.CFG and $2:PATH of INIT_GOALS.CFG
$ source src/gazebo/robot_gazebo/scripts/bringup_all.bash
# Quit by
$ killall screen
# MAPF node, each agent needs one
$ rosrun mapf_pkg mapf.py -h
usage: mapf.py [-h] [--current-robot-num CURRENT_ROBOT_NUM]
               [--robots-num ROBOTS_NUM] [--robot-diameter ROBOT_DIAMETER]
               [--map-resolution MAP_RESOLUTION]

Reinforcement Learning Arguments

optional arguments:
  -h, --help            show this help message and exit
  --current-robot-num CURRENT_ROBOT_NUM
                        The number of current robot. (For env) (default: None)
  --robots-num ROBOTS_NUM
                        The amount of all robots. (For env) (default: None)
  --robot-diameter ROBOT_DIAMETER
                        The diameter of robot (default: 0.25, according to
                        TB3) (For env) (default: 0.25)
  --map-resolution MAP_RESOLUTION
                        The resolution of map (default: 0.01) (For env)
                        (default: 0.01)
```
