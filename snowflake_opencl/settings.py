from __future__ import print_function

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class Settings(object):
    def __init__(self,
                 use_local_mem,
                 use_plane_offsets,
                 enqueue_iterations,
                 use_local_register,
                 unroll_kernel,
                 force_local_work_size,
                 label
                 ):
        self.use_local_mem = use_local_mem
        self.use_plane_offsets = use_plane_offsets
        self.enqueue_iterations = enqueue_iterations
        self.use_local_register = use_local_register
        self.unroll_kernel = unroll_kernel
        self.force_local_work_size = force_local_work_size
        self.label = label