from __future__ import print_function

import argparse

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class Settings(object):
    def __init__(self,
                 use_local_mem,
                 use_plane_offsets,
                 enqueue_iterations,
                 use_local_register,
                 unroll_kernel,
                 force_local_work_size,
                 remove_for_body_fence,
                 pencil_kernel_size_threshold,
                 label
                 ):
        self.use_local_mem = use_local_mem
        self.use_plane_offsets = use_plane_offsets
        self.enqueue_iterations = enqueue_iterations
        self.use_local_register = use_local_register
        self.unroll_kernel = unroll_kernel
        self.force_local_work_size = force_local_work_size
        self.remove_for_body_fence = remove_for_body_fence
        self.pencil_kernel_size_threshold = pencil_kernel_size_threshold
        self.label = label

    @staticmethod
    def add_settings_parsers(parser):
        parser.add_argument("size", type=int, help="mesh edge size")
        parser.add_argument("-t", "--test-method", type=str, default="none")
        parser.add_argument("-rnp", "--run-no-pencil", action="store_true")
        parser.add_argument("-i", "--iterations", type=int, default=1)
        parser.add_argument("-lm", "--use-local-mem", action="store_true")
        parser.add_argument("-lr", "--use-local-register", action="store_true")
        parser.add_argument("-po", "--use-plane-offsets", action="store_true")
        parser.add_argument("-sm", "--show-mesh", action="store_true")
        parser.add_argument("-uk", "--unroll-kernel", action="store_true")
        parser.add_argument("-ff", "--force-float", action="store_true")
        parser.add_argument("-sgc", "--show-generated-code", action="store_true")
        parser.add_argument("-sop", "--set-operator", type=str,
                            help='one of 7pt, 13pt, 3x3x3, 5x5x5')
        parser.add_argument("-flws", "--force-local-work-size", type=str)
        parser.add_argument("-tl", "--timer-label", type=str, default="opencl pencil test")
        parser.add_argument("-ei", "--enqueue-iterations", type=int)
        parser.add_argument("-pkst", "--pencil-kernel-size-threshold", type=int)
        parser.add_argument("-rfbf", "--remove-for-body-fence", action="store_true")
