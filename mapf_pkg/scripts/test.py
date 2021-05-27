#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ContactsState

def callback(data):
    print("Length {}".format(len(data.states)))
    for i, e in enumerate(data.states):
        print("Pair {}: {} <---> {}".format(i, e.collision1_name, e.collision2_name))
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/robot0/bumper", ContactsState, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
