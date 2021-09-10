import time
import signal
import sys
import numpy as np

def signal_handler(sig, frame):
    print("This works!")
    sys.exit(0)

class myClass:
    """
    Example class to test if python class can save an attribute when given the
    keyboard interrupt signal. It works but if we want to save an object stored
    in C++ it hangs.
    """

    def __init__(self):
        # Define some variables
        self.var = np.array([4, 5, 6, 7])
        self.setup()

    def setup(self):
        signal.signal(signal.SIGINT, self.catch)

    def catch(self, sig, frame):
        np.savetxt("data.txt", self.var)
        sys.exit(0)

if __name__ == '__main__':
    mine = myClass()
    time.sleep(50)
