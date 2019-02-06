import g2o
import numpy as np

class Pose:

    def __init__(self, *args):
        self.iso = Pose(*args)

    def __getstate__(self):
        state = {}
        state['matrix'] = self.iso.matrix()
        return state

    def __setstate__(self, state):
        self.iso = Pose(state['matrix'])

    def position(self):
        return self.iso.position()

    def orientation(self):
        return self.iso.orientation()

    def matrix(self):
        return self.iso.matrix()

    def inverse(self):
        return self.iso.inverse()


import dill

q = Pose)
print(q.position())
print(q.orientation())

with open('test.pkl', 'wb') as f:
    dill.dump(q, f)

with open('test.pkl', 'rb') as f:
    l = dill.load(f)
    print(l.matrix())




