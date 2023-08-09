from scipy.io import savemat
import numpy as np

class ArrayRecorder(object):

    def __init__(self):
        self.arrays = {}

    def add(self, key: str, array: np.array):
        if key in self.arrays:
            msg = f"{self.__class__.__name__} add key: {key} already has record."
            raise RuntimeError(msg)
        self.arrays[key] = np.array(array)

    def to_mat(self, out_f: str):
        savemat(out_f, self.arrays, oned_as="row")
