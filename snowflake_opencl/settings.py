from __future__ import print_function

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class Settings(object):
    def __init__(self, use_local_mem, use_plane_offsets):
        self.use_local_mem = use_local_mem
        self.use_plane_offsets = use_plane_offsets