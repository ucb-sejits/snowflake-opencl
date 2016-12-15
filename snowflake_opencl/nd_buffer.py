import pycl as cl

# noinspection PyProtectedMember

__author__ = 'chick markley, seunghwan choi'


class NDBuffer(object):
    def __init__(self, queue, ary, blocking=True):
        self.ary = ary
        self.shape = ary.shape
        self.dtype = ary.dtype
        self.ndim = ary.ndim
        self.queue = queue
        self.buffer, evt = cl.buffer_from_ndarray(queue, ary)
        if blocking:
            evt.wait()
