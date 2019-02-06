import g2o
import numpy as np

class Pose:
    """
    Wrapper class for Pose with pickling support
    """

    def __init__(self, *args):
        self.iso = g2o.Isometry3d(*args)

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

    def __mul__(self, other):
        return self.iso * other