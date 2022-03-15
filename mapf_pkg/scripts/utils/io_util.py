def read_file(f):
    l = list()
    with open(f) as file:
        while (line := file.readline().rstrip()):
            l.append(tuple(map(float, line.rstrip().split(' '))))
    return l