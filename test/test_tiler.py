from __future__ import print_function
import unittest
import numpy as np

from snowflake.compiler_nodes import IterationSpace, NDSpace, Space
from snowflake.nodes import RectangularDomain

from snowflake_opencl.ocl_tiler import OclTiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestTiler(unittest.TestCase):
    def test_index_generation_2d(self):
        base_shape = (11, 16)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, -1, 1), (1, -1, 1)))
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, context=DeviceMock(max_work_item_sizes=[5, 6, 1]),
                         force_local_work_size=(5, 6))

        self.assertEqual(tiler.reference_shape, base_shape)
        self.assertEqual(tiler.dimensions, 2)
        self.assertEqual(tiler.packed_iteration_shape, (9, 14))
        self.assertEqual(tiler.local_work_size, (5, 6))
        self.assertEqual(tiler.tiling_shape, (2, 3))

        self.assertEqual(tiler.global_size_1d, 10 * 18)
        self.assertEqual(tiler.local_work_size_1d, 30)

        self.assertEqual(tiler.get_tile_number(0), 0)
        self.assertEqual(tiler.get_tile_number(29), 0)
        self.assertEqual(tiler.get_tile_number(30), 1)
        self.assertEqual(tiler.get_tile_number(59), 1)

        self.assertEqual(tiler.get_tile_coordinates(0), (0, 0))
        self.assertEqual(tiler.get_tile_coordinates(30), (0, 1))
        self.assertEqual(tiler.get_tile_coordinates(59), (0, 1))
        self.assertEqual(tiler.get_tile_coordinates(60), (0, 2))
        self.assertEqual(tiler.get_tile_coordinates(79), (0, 2))

        self.assertEqual(tiler.get_tile_coordinates(90), (1, 0))
        self.assertEqual(tiler.get_tile_coordinates(119), (1, 0))

        mesh_1d_coords = np.zeros(base_shape)
        mesh_tile_numbers = np.zeros(base_shape)

        # coord = tiler.global_index_to_coord(63)
        print("is valid(63) {}".format(tiler.valid_1d_index(63)))
        for index_1d in range(tiler.global_size_1d):
            coord = tiler.global_index_to_coord(index_1d)
            if tiler.valid_1d_index(index_1d):
                print("index_1d {} coord {}".format(index_1d, coord))
                mesh_1d_coords[coord] = index_1d
                mesh_tile_numbers[coord] = tiler.get_tile_number(index_1d)
            else:
                print("Out of bound coordinate at {}".format(coord))

        print("space with tile numbers")
        for i in range(base_shape[0]-1, -1, -1):
            for j in range(base_shape[1]):
                print("{:5d}".format(int(mesh_tile_numbers[(i, j)])), end="")
            print()
        print("space with global_id_numbers")
        for i in range(base_shape[0]-1, -1, -1):
            for j in range(base_shape[1]):
                print("{:5d}".format(int(mesh_1d_coords[(i, j)])), end="")
            print()
        self.assertEqual(tiler.global_index_to_coord(0), (1, 1))

    def test_2d_11x16(self):
        base_shape = (11, 16)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, -1, 1), (1, -1, 1)))
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, context=DeviceMock(max_work_item_sizes=[5, 6, 1]),
                         force_local_work_size=(5, 6))

        self.assertEqual(tiler.reference_shape, base_shape)
        self.assertEqual(tiler.dimensions, 2)
        self.assertEqual(tiler.packed_iteration_shape, (9, 14))
        self.assertEqual(tiler.local_work_size, (5, 6))
        self.assertEqual(tiler.tiling_shape, (2, 3))

    def test_2d_130x514(self):
        base_shape = (130, 514)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 129, 1), (1, 513, 1)))
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, context=DeviceMock())

        self.assertEqual(tiler.reference_shape, base_shape)
        self.assertEqual(tiler.dimensions, 2)
        self.assertEqual(tiler.packed_iteration_shape, (128, 512))
        self.assertEqual(tiler.local_work_size, (16, 32))
        self.assertEqual(tiler.tiling_shape, (8, 16))

    def test_2d_130x514_strided(self):
        base_shape = (130, 514)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 128, 2), (2, 513, 2))),
            RectangularDomain(((2, 129, 2), (1, 512, 2))),
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, context=DeviceMock())

        self.assertEqual(tiler.reference_shape, base_shape)
        self.assertEqual(tiler.dimensions, 2)
        self.assertEqual(tiler.packed_iteration_shape, (64, 256))
        self.assertEqual(tiler.local_work_size, (16, 32))
        self.assertEqual(tiler.tiling_shape, (4, 8))

    def test_iteration_space_factory(self):
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 10, 1), (1, 10, 1)))
        ])

        self.assertEqual(len(space1.space.spaces), 1)
        self.assertEqual(space1.space.spaces[0].low, (1, 1))
        self.assertEqual(space1.space.spaces[0].high, (10, 10))
        self.assertEqual(space1.space.spaces[0].stride, (1, 1))

        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 10, 2), (2, 11, 2))),
            RectangularDomain(((2, 11, 2), (1, 10, 2))),
        ])

        self.assertEqual(len(space1.space.spaces), 2)
        self.assertEqual(space1.space.spaces[0].low, (1, 2))
        self.assertEqual(space1.space.spaces[0].high, (10, 11))
        self.assertEqual(space1.space.spaces[0].stride, (2, 2))
        self.assertEqual(space1.space.spaces[1].low, (2, 1))
        self.assertEqual(space1.space.spaces[1].high, (11, 10))
        self.assertEqual(space1.space.spaces[1].stride, (2, 2))


class DeviceMock(object):
    def __init__(self, max_work_item_sizes=[512, 512, 512], max_work_group_size=512, max_compute_units=40):
        self.max_work_item_sizes = max_work_item_sizes
        self.max_work_group_size = max_work_group_size
        self.max_compute_units = max_compute_units


class IterationSpaceFactory(object):
    @staticmethod
    def get(rectangular_domains):
        """
        build an iteration space with an empty body for tests
        :param rectangular_domains: A list of RectangularDomains
        :return:
        """
        return IterationSpace(
            space=NDSpace(
                [
                    Space(domain.lower, domain.upper, domain.stride)
                    for domain in rectangular_domains
                    ]
            ),
            body=[],
        )
