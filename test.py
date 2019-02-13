
import multiprocessing
import time
import cv2
import pickle

class Test:
    def __init__(self):
        self.data = cv2.line_descriptor.BinaryDescriptor_createBinaryDescriptor()

t = Test()
print(t.data)

def func(test):
    print(test.data)

p = multiprocessing.Process(target=func, args=(t,))
p.start()
p.join()
