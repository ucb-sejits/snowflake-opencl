from __future__ import print_function

import operator

from snowflake_opencl.local_size_computer import LocalSizeComputer

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class OclTiler(object):
    """
    figure out how to tile an iteration space given the underlying mesh in which
    the iteration space exists, figure out how to tile that space in an OpenCL
    appropriate style.  Linearize the various components to 1-d for now, so the
    The compiler using this can work in arbitrary numbers of dimensions
    """
    # TODO: Implement bounds checking for when local_work_size does not divide space evenly
    # TODO: Still need a more elegant way to manage multi-domain iteration spaces
    def __init__(self, reference_shape, iteration_space, context=None, force_local_work_size=None):
        self.reference_shape = reference_shape
        self.iteration_space = iteration_space
        self.context = context
        self.dimensions = len(self.reference_shape)

        self.packed_iteration_shape = self.get_packed_shape()
        self.local_work_size = force_local_work_size if force_local_work_size is not None \
            else LocalSizeComputer(self.packed_iteration_shape, self.context).compute_local_size_bulky()
        self.tiling_shape = self.get_tiling_shape()
        self.tiling_divisors = self.get_tiling_divisors()
        self.local_divisors = self.get_local_divisors()

        self.virtual_global_size = tuple(x * y for x, y in zip(self.local_work_size, self.tiling_shape))
        self.global_size_1d = reduce(operator.mul, self.virtual_global_size, 1)
        self.local_work_size_1d = reduce(operator.mul, self.local_work_size, 1)

    def global_index_to_coord(self, index_1d, iteration_space_index=0):
        tile_coord = self.get_tile_coordinates(index_1d)
        local_coord = self.get_local_coordinates(index_1d)

        coord = []
        for dim in range(self.dimensions):
            coord.append(
                tile_coord[dim] * self.local_work_size[dim] +
                local_coord[dim] +
                self.iteration_space.space.spaces[iteration_space_index].low[dim])

        return tuple(coord)

    def get_tile_number(self, index_1d):
        """
        return a one dimensional tiler number from a 1 dimension index in the iteration space
        :param index_1d:
        :return:
        """
        return int(index_1d / self.local_work_size_1d)

    def get_tile_coordinates(self, index_1d):
        """
        return an n-dimensional coordinate of a given tile in the space
        :param index_1d:
        :return:
        """
        tile_number = self.get_tile_number(index_1d)
        coord = []
        for dim in range(self.dimensions):
            coord.append(int(tile_number / self.tiling_divisors[dim]))
            tile_number = tile_number % self.tiling_divisors[dim]

        return tuple(coord)

    def get_tiling_divisors(self):
        return [
            reduce(operator.mul, self.tiling_shape[(dim+1):], 1)
            for dim in range(self.dimensions)
            ]

    def get_local_coordinates(self, index_1d):
        internal_1d = index_1d % self.local_work_size_1d

        coord = []
        for dim in range(self.dimensions):
            coord.append(int(internal_1d / self.local_divisors[dim]))
            internal_1d = internal_1d % self.local_divisors[dim]

        return tuple(coord)

    def get_local_divisors(self):
        return [
            reduce(operator.mul, self.local_work_size[(dim+1):], 1)
            for dim in range(self.dimensions)
            ]

    def get_tiling_shape(self):
        tiling_shape = tuple(
            [
                int((self.packed_iteration_shape[dim]-1) / self.local_work_size[dim]) + 1
                for dim in range(self.dimensions)
                ])
        return tiling_shape

    def get_packed_shape(self):
        """
        compact the iteration space by fixing highs and lows as necessary and then
        squishing out the strides.
        IMPORTANT: Iterations may contain multiple spaces, currently these must all be the
        same size so that they can all be run in the same kernel

        :return: a single packed shape for all the iteration spaces
        """
        packed_shapes = []

        def make_low(floor, dimension):
            return floor if floor >= 0 else self.reference_shape[dimension] + floor

        def make_high(ceiling, dimension):
            return ceiling if ceiling > 0 else self.reference_shape[dimension] + ceiling

        for space in self.iteration_space.space.spaces:
            lows = tuple(make_low(low, dim) for dim, low in enumerate(space.low))
            highs = tuple(make_high(high, dim) for dim, high in enumerate(space.high))
            strides = space.stride
            packed_shapes.append(
                tuple(
                    [(high - low + stride - 1) / stride
                     for (low, high, stride) in list(zip(lows, highs, strides))
                     ]
                ))

        if all(other_shapes == packed_shapes[0] for other_shapes in packed_shapes[1:]):
            return packed_shapes[0]
        else:
            raise NotImplementedError("Different number of threads per space in IterationSpace not implemented.")
