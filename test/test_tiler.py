from __future__ import print_function
import unittest

from snowflake_opencl.local_size_computer import LocalSizeComputer

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestTiler(unittest.TestCase):
    def test_work_sizes(self):
        print(" ".join("{:10d}".format(j) for j in range(30, 101, 3)))
        for i in range(30, 101, 3):
            print("{:7d}   ".format(i), end="")
            for j in range(30, 101, 3):

                local_size_computer = LocalSizeComputer((i, j))

                print("{:10s}".format(local_size_computer.compute_local_size_bulky().__str__()), end="")
            print()

        print(" ".join("{:10d}".format(j) for j in range(30, 101, 3)))
        for i in range(30, 101, 3):
            print("{:7d}   ".format(i), end="")
            for j in range(30, 101, 3):

                local_size_computer = LocalSizeComputer((i, j))

                print("{:10s}".format(local_size_computer.compute_local_size_thin().__str__()), end="")
            print()
