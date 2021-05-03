import math

# dist acheived in Python 3.8 officially
def dist(p, q):
    return math.sqrt(sum((px - qx)**2 for px, qx in zip(p, q)))