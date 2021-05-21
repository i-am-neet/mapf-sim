# System
Name                        | Version      
----------------------------|-------:
Ubuntu                      | 20.04
Robot Operating System (ROS)| noetic
Python                      | 3.6.13
Pip                         | 20.0.1
Nvidia driver               | 450.119.03
CUDA                        | 10.1
cUDNN                       | 7.6

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
```

Build Stage from source
```bash
$ cd $HOME
$ mkdir stage4
$ cd stage4
$ git clone git://github.com/rtv/Stage.git
# or https://github.com/CodeFinder2/Stage.git
$ export STG=$HOME/stg
$ cmake -DCMAKE_INSTALL_PREFIX=$STG Stage
$ make
$ make install
$ export LD_LIBRARY_PATH=$STG/lib
# Stage LD
echo "export STG=$HOME/stg" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$STG/lib" >> ~/.bashrc
echo "export PATH=$PATH:$STG/bin" >> ~/.bashrc
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
$ source src/robot_stage/scripts/bringup_all.bash
# Quit by
$ source src/robot_stage/scripts/quit_all.bash
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
