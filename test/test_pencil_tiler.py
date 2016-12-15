from __future__ import print_function
import unittest
import numpy as np

from snowflake.compiler_nodes import IterationSpace, NDSpace, Space
from snowflake.nodes import RectangularDomain, StencilComponent, WeightArray

from snowflake_opencl.ocl_3d_pencil_tiler import Ocl3DPencilTiler
from snowflake_opencl.ocl_tiler import OclTiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestPencilTiler(unittest.TestCase):
    """
    The pencil tiler works only in 3 dimensions right now.
    It creates tiles in the 2nd and 3rd dimensions (i.e. space closest to the unit stride)
    then has the kernel iterate over the 1st dimension.
    The processing is done in local memory, this requires a series of planes
    The number of planes is determined by the shadow of the stencil in the 1st dimension

    It is possible that multiple meshes will be used during the stencil computation,
    the local memory is used here only for the primary mesh.

    Many different indices will be generated in order to create the opencl kernel.
    These indices will be computed from functions of the global_id and the local_id
    """
    def test_index_simple_jacobi_32_cubed(self):
        underlying_shape = (34, 34, 34)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, -1, 1), (1, -1, 1), (1, -1, 1)))
        ])
        sc = StencilComponent(
            'mesh',
            WeightArray([
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, -6, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ])
        )

        tiler = Ocl3DPencilTiler(underlying_shape, iteration_space=space1, stencil=sc, context=DeviceMock(max_work_item_sizes=[5, 6, 1]))

        self.assertEqual(tiler.underlying_shape, underlying_shape)
        self.assertEqual(tiler.dimensions, 3)

        self.assertEqual(tiler.stencil.weights.shape, (3, 3, 3))
        self.assertEqual(tiler.inner_tiled_shape, (32, 32))
        self.assertEqual(tiler.packed_iteration_shape, (32, 32))

    def test_index_gsrb_32_cubed(self):
        underlying_shape = (34, 34, 34)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, -2, 2), (1, -2, 2))),
            RectangularDomain(((2, -1, 2), (2, -1, 2)))
        ])
        sc = StencilComponent(
            'mesh',
            WeightArray([
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, -6, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ])
        )

        tiler = Ocl3DPencilTiler(underlying_shape, iteration_space=space1, stencil=sc, context=DeviceMock(max_work_item_sizes=[5, 6, 1]))

        self.assertEqual(tiler.underlying_shape, underlying_shape)
        self.assertEqual(tiler.dimensions, 3)

        self.assertEqual(tiler.stencil.weights.shape, (3, 3, 3))
        self.assertEqual(tiler.tiled_shape, (32, 32))

        # self.assertEqual(tiler.packed_iteration_shape, (9, 14))
        # self.assertEqual(tiler.local_work_size, (5, 6))
        # self.assertEqual(tiler.tiling_shape, (2, 3))
        #
        # self.assertEqual(tiler.global_size_1d, 10 * 18)
        # self.assertEqual(tiler.local_work_size_1d, 30)
        #
        # self.assertEqual(tiler.get_tile_number(0), 0)
        # self.assertEqual(tiler.get_tile_number(29), 0)
        # self.assertEqual(tiler.get_tile_number(30), 1)
        # self.assertEqual(tiler.get_tile_number(59), 1)
        #
        # self.assertEqual(tiler.get_tile_coordinates(0), (0, 0))
        # self.assertEqual(tiler.get_tile_coordinates(30), (0, 1))
        # self.assertEqual(tiler.get_tile_coordinates(59), (0, 1))
        # self.assertEqual(tiler.get_tile_coordinates(60), (0, 2))
        # self.assertEqual(tiler.get_tile_coordinates(79), (0, 2))
        #
        # self.assertEqual(tiler.get_tile_coordinates(90), (1, 0))
        # self.assertEqual(tiler.get_tile_coordinates(119), (1, 0))
        #
        # mesh_1d_coords = np.zeros(underlying_shape)
        # mesh_tile_numbers = np.zeros(underlying_shape)
        #
        # # coord = tiler.global_index_to_coord(63)
        # print("is valid(63) {}".format(tiler.valid_1d_index(63)))
        # for index_1d in range(tiler.global_size_1d):
        #     coord = tiler.global_index_to_coord(index_1d)
        #     if tiler.valid_1d_index(index_1d):
        #         # print("index_1d {} coord {}".format(index_1d, coord))
        #         mesh_1d_coords[coord] = index_1d
        #         mesh_tile_numbers[coord] = tiler.get_tile_number(index_1d)
        #     else:
        #         # print("Out of bound coordinate at {}".format(coord))
        #         pass
        #
        # print("space with tile numbers")
        # for i in range(underlying_shape[0]-1, -1, -1):
        #     for j in range(underlying_shape[1]):
        #         print("{:5d}".format(int(mesh_tile_numbers[(i, j)])), end="")
        #     print()
        # print("space with global_id_numbers")
        # for i in range(underlying_shape[0]-1, -1, -1):
        #     for j in range(underlying_shape[1]):
        #         print("{:5d}".format(int(mesh_1d_coords[(i, j)])), end="")
        #     print()
        # self.assertEqual(tiler.global_index_to_coord(0), (1, 1))

    def test_2d_11x16(self):
        base_shape = (11, 16)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, -1, 1), (1, -1, 1)))
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, device=DeviceMock(max_work_item_sizes=[5, 6, 1]),
                         force_local_work_size=(5, 6))

        self.assertEqual(tiler.underlying_shape, base_shape)
        self.assertEqual(tiler.dimensions, 2)
        self.assertEqual(tiler.packed_iteration_shape, (9, 14))
        self.assertEqual(tiler.local_work_size, (5, 6))
        self.assertEqual(tiler.tiling_shape, (2, 3))

    def test_2d_130x514(self):
        base_shape = (130, 514)
        space1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 129, 1), (1, 513, 1)))
        ])

        tiler = OclTiler(base_shape, iteration_space=space1, device=DeviceMock())

        self.assertEqual(tiler.underlying_shape, base_shape)
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

        tiler = OclTiler(base_shape, iteration_space=space1, device=DeviceMock())

        self.assertEqual(tiler.underlying_shape, base_shape)
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

    def test_packed_shape_generator(self):
        base_shape = (52, 52)
        space_1 = IterationSpaceFactory.get([RectangularDomain(((1, 50, 2), (1, 50, 2)))])
        tiler = OclTiler(base_shape, iteration_space=space_1, device=DeviceMock())
        print("tiler packed shape{}".format(tiler.get_packed_shape()))
        print("tiler local_reference_shape {}".format(tiler.calculate_local_reference_array_shape()))
        print("tiler iteration space shape {}".format(tiler.iteration_space_shape))

        base_shape = (52, 52)
        space_1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 49, 2), (1, 49, 2))),
            RectangularDomain(((2, 50, 2), (2, 50, 2))),
        ])
        tiler = OclTiler(base_shape, iteration_space=space_1, device=DeviceMock())
        print("Base shape {}".format(base_shape))
        print("space {}".format(space_1))
        print("tiler packed shape{}".format(tiler.get_packed_shape()))
        print("tiler local_reference_shape {}".format(tiler.calculate_local_reference_array_shape()))
        print("tiler iteration space shape {}".format(tiler.iteration_space_shape))

    def test_red_space(self):
        """
        test the combined rectangular domain that constitutes the red space
        of a GSRB in-place iteration
        :return:
        """
        base_shape = (6, 6)
        space_1 = IterationSpaceFactory.get([
            RectangularDomain(((1, 3, 2), (1, 3, 2))),
            RectangularDomain(((2, 4, 2), (2, 4, 2))),
        ])
        tiler = OclTiler(base_shape, iteration_space=space_1, device=DeviceMock())
        print("Base shape {}".format(base_shape))
        print("space {}".format(space_1))
        print("tiler packed shape{}".format(tiler.get_packed_shape()))
        print("tiler local_reference_shape {}".format(tiler.calculate_local_reference_array_shape()))
        print("tiler iteration space shape {}".format(tiler.iteration_space_shape))


class DeviceMock(object):
    def __init__(self, max_work_item_sizes=None, max_work_group_size=512, max_compute_units=40):
        if max_work_item_sizes is None:
            max_work_item_sizes = [512, 512, 512]
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
