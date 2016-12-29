from __future__ import print_function

import pycl as cl

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class NDBuffer(object):
    def __init__(self, queue, ary, blocking=True):
        self.ary = ary
        self.shape = ary.shape
        self.dtype = ary.dtype
        self.ndim = ary.ndim
        self.buffer, evt = cl.buffer_from_ndarray(queue, ary)
        if blocking:
            evt.wait()
