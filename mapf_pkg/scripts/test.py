import sys
import rospy
from mapf_pkg.srv import *
from mapf_pkg.msg import float1d, float2d

def add_two_ints_client(ng):
    rospy.wait_for_service('change_goals')
    mm = float2d()
    for i, e in enumerate(ng):
        m = float1d(e)
        mm.data.append(m)
    try:
        add_two_ints = rospy.ServiceProxy('change_goals', ChangeGoals)
        resp1 = add_two_ints(mm)
        return resp1.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    # ng = [(1,2,3),(4,5,6),(7,8,9),(0,0,0)]
    ng = [[ 1.,2.,3. ],[ 4.,5.,6. ],[ 7.,8.,9. ],[ 1.,1.,1. ]]
    # ng = [1.,2.,3.,4.,5.]
    print("Requesting %s"%(ng))
    print("%s"%(add_two_ints_client(ng)))