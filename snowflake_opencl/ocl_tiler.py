from __future__ import print_function

import operator

from snowflake_opencl.local_size_computer import LocalSizeComputer
from ctree.c.nodes import Mod, Div, Constant, Add, Mul, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile, LtE, And, If


__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class OclTiler(object):
    """
    This computes and manages the descriptive information for an iteration space
    and its underlying mesh.
    It computes a tiling over the iteration space appropriate for the supplied opencl device
    It provides coordinate transformations as AST snippets that convert from 1-d to n-d
    for the following regions:

    The iteration space may be composed of multiple rectangular domains, but it must
    satisfy that each rectangular domain must be within (stride + 1) of at least one other
    domain.  This function does not check this condition but assumes it has been satisfied
    by snowflakes analysis engine
    """
    # TODO: Still need a more elegant way to manage multi-domain iteration spaces
    # TODO: Make striding work correctly
    # TODO: Confirm that guards work correctly with striding
    def __init__(self, underlying_shape, iteration_space, device=None, force_local_work_size=None):
        """

        :param underlying_shape:
        :param iteration_space:
        :param device:
        :param force_local_work_size:
        :return:
        """
        self.underlying_shape = underlying_shape
        self.iteration_space = iteration_space
        self.device = device
        self.dimensions = len(self.underlying_shape)

        self.iteration_space_shape = self.get_iteration_space_shape()
        self.packed_iteration_shape = self.get_packed_shape()
        self.local_work_size = force_local_work_size if force_local_work_size is not None \
            else LocalSizeComputer(self.packed_iteration_shape, self.device).compute_local_size_bulky()
        self.tiling_shape = self.get_tiling_shape()
        self.tiling_divisors = self.get_tiling_divisors()
        self.local_divisors = self.get_local_divisors()
        self.local_reference_shape = self.calculate_local_reference_array_shape()

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

    def add_guards_if_necessary(self, node):
        guards = []
        for dim in range(self.dimensions):
            if self.packed_iteration_shape[dim] % self.local_work_size[dim] != 0:
                guards.append(LtE(SymbolRef("index_{}".format(dim)), Constant(self.packed_iteration_shape[dim])))

        if len(guards) > 0:
            conditional = guards[0]
            for additional_term in guards[1:]:
                conditional = And(conditional, additional_term)

            return If(conditional, node)
        else:
            return node

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

    def valid_1d_index(self, index_1d, iteration_space_index=0):
        tile_coord = self.get_tile_coordinates(index_1d)
        local_coord = self.get_local_coordinates(index_1d)

        iteration_space_coord = tuple((x * y) + z for x, y, z in zip(tile_coord, self.local_work_size, local_coord))
        return all(x < y for x, y in zip(iteration_space_coord, self.packed_iteration_shape))

    def global_index_to_coordinate_expressions(self, global_index_symbol, iteration_space_index=0):
        tile_coords = self.get_tile_coordinates_expression(global_index_symbol)
        local_coords = self.get_local_coordinates_expression((global_index_symbol))

        coords = []
        for dim in range(self.dimensions):
            coords.append(
                Add(
                    Mul(
                        Add(
                            Mul(tile_coords[dim], Constant(self.local_work_size[dim])),
                            local_coords[dim]
                        ),
                        Constant(self.iteration_space.space.spaces[iteration_space_index].stride[dim])
                    ),
                    Constant(self.iteration_space.space.spaces[iteration_space_index].low[dim])
                )
            )
        return coords

    def get_tile_number_expression(self, index_1d_symbol):
        tile_number = Div(index_1d_symbol, Constant(self.local_work_size_1d))
        tile_number._force_parentheses = True
        return tile_number

    def get_tile_coordinates_expression(self, index_1d_symbol):
        tile_number = self.get_tile_number_expression(index_1d_symbol)
        coords = []
        for dim in range(self.dimensions):
            coord = Div(tile_number, Constant(self.tiling_divisors[dim]))
            coord._force_parentheses = True
            tile_number = Mod(tile_number, Constant(self.tiling_divisors[dim]))
            tile_number._force_parentheses = True
            coords.append(coord)

        return coords

    def get_local_coordinates_expression(self, index_1d_symbol):
        internal_1d = Mod(index_1d_symbol, Constant(self.local_work_size_1d))
        internal_1d._force_parentheses = True

        coords = []
        for dim in range(self.dimensions):
            coord = Div(internal_1d, Constant(self.local_divisors[dim]))
            coord._force_parentheses = True
            internal_1d = Mod(internal_1d, Constant(self.local_divisors[dim]))
            internal_1d._force_parentheses = True
            coords.append(coord)

        return tuple(coords)

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
            return floor if floor >= 0 else self.underlying_shape[dimension] + floor

        def make_high(ceiling, dimension):
            return ceiling if ceiling > 0 else self.underlying_shape[dimension] + ceiling

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

    def get_iteration_space_shape(self):
        """
        return the smallest space that is the rectangular union of the
        supplied interation_space

        :return: a single shape for all the iteration spaces
        """
        packed_shapes = []

        def make_low(floor, dimension):
            return floor if floor >= 0 else self.underlying_shape[dimension] + floor

        def make_high(ceiling, dimension):
            return ceiling if ceiling > 0 else self.underlying_shape[dimension] + ceiling

        spaces = self.iteration_space.space.spaces
        lows = [make_low(low, dim) for dim, low in enumerate(spaces[0].low)]
        highs = [make_high(high, dim) for dim, high in enumerate(spaces[0].high)]

        for space in spaces[1:]:
            lows = [min(lows[dim],make_low(low, dim)) for dim, low in enumerate(space.low)]
            highs = [max(highs[dim],make_high(high, dim)) for dim, high in enumerate(space.high)]

        return tuple((highs[dim] - lows[dim]) + 1 for dim in range(len(self.underlying_shape)))

    def calculate_local_reference_array_shape(self):
        ghost_size = self.underlying_shape[0] - self.packed_iteration_shape[0]
        local_reference_array_shape = []
        for x in self.local_work_size:
            local_reference_array_shape.append(x + ghost_size)
        return local_reference_array_shape
