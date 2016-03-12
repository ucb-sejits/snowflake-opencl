from __future__ import print_function
import unittest

from snowflake.compiler_nodes import Space

from snowflake_opencl.local_size_computer import LocalSizeComputer
from snowflake_opencl.tiler import Tiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestTiler(unittest.TestCase):
    def test_basic_size(self):
        space = Space(high=10, )
        tiler = Tiler((10,), (8,), (1,))
        self.assertEqual(tiler.complete_1d_size, 10)
        self.assertEqual(tiler.iteration_1d_size, 8)
        self.assertEqual(tiler.offset_1d_size, 1)

        tiler = Tiler((10, 12), (8, 4), (1, 3))
        self.assertEqual(tiler.complete_1d_size, 120)
        self.assertEqual(tiler.iteration_1d_size, 32)
        self.assertEqual(tiler.offset_1d_size, 3)

        tiler = Tiler((10, 10, 14), (8, 8, 2), (1, 1, 1))
        self.assertEqual(tiler.complete_1d_size, 1400)
        self.assertEqual(tiler.iteration_1d_size, 128)
        self.assertEqual(tiler.offset_1d_size, 1)

    def test_local_work_size(self):
        tiler = Tiler(complete_space=(10, 10), iteration_space=(8, 8), offset=(1, 1))

        self.assertEqual(tiler.relative_iteration_space, (8, 8))


    def test_work_sizes(self):
        tiler = Tiler((100, 100), (77, 77), (1, 1))

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
