#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ContactsState

c = 0
current_number = input("Input robot number: ")

def callback(data):
    global c
    # print("Length {}".format(len(data.states)))
    for i, e in enumerate(data.states):
        # print("Pair {}: {} <---> {}".format(i, e.collision1_name, e.collision2_name))
        A = [e.collision1_name, e.collision2_name]

        if any('ground_plane' in a.lower() for a in A):
            break
        elif any('wall' in a.lower() for a in A):
            print("{} Hit the wall!!!!!".format(c))
            c+=1
        elif all('robot' in a.lower() for a in A):
            print("{} Hit other robot!!".format(c))
            c+=1
        else:
            print(A[0])
            print("{} Other condition????".format(c))
            print(A[1])
            print()
            c+=1
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/robot{}/bumper".format(input("Input : ")), ContactsState, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
