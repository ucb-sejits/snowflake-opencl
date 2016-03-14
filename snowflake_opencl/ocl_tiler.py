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
    def __init__(self, reference_shape, iteration_space, context=None, force_local_work_size=None):
        self.reference_shape = reference_shape
        self.iteration_space = iteration_space
        self.context = context
        self.dimensions = len(self.reference_shape)

        self.packed_iteration_shape = self.get_packed_shape()
        self.local_work_size = force_local_work_size if force_local_work_size is not None \
            else LocalSizeComputer(self.packed_iteration_shape, self.context).compute_local_size_bulky()
        self.tiling_shape = self.get_tiling_shape()

        self.global_size_1d = reduce(operator.mul, self.packed_iteration_shape, 1)
        self.local_work_size_1d = reduce(operator.mul, self.local_work_size, 1)

    def global_index_to_coord(self, global_index):
        return 0

    def get_tile_number(self, index_1d):
        return int(index_1d / self.local_work_size_1d)

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
