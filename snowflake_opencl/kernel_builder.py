import ctypes
import operator

from ctree.c.nodes import Constant, SymbolRef, Assign, FunctionCall, \
    ArrayRef, Add, Mod, Mul, Div, Lt, If, For, PostInc
from ctree.c.nodes import MultiNode
from ctree.templates.nodes import StringTemplate

# noinspection PyProtectedMember
from snowflake.stencil_compiler import CCompiler
import math

from snowflake_opencl.local_size_computer import LocalSizeComputer
from snowflake_opencl.ocl_tiler import OclTiler
from snowflake_opencl.primary_mesh_to_plane_transformer import PrimaryMeshToPlaneTransformer
from snowflake_opencl.primary_mesh_to_register_transformer import PrimaryMeshToRegisterTransformer

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


class KernelBuilder(CCompiler.IterationSpaceExpander):
    def __init__(self, index_name, reference_array_shape, stencil, context, device, settings):
        self.stencil = stencil
        self.stencil_node = stencil.op_tree
        self.ghost_size = tuple((x - 1)/2 for x in self.stencil_node.weights.shape)
        self.context = context
        self.device = device
        self.use_doubles = "cl_khr_fp64" in self.device.extensions
        self.use_local_mem = settings.use_local_mem
        self.use_local_register = settings.use_local_register
        self.settings = settings

        self.planes = self.stencil_node.weights.shape[0]

        # compute_plane_info sets the following instance vars
        self.plane_size = None
        self.plane_size_1d = None
        self.local_work_size = None
        self.local_work_size_1d = None

        self.global_work_size = None
        self.global_work_size_1d = None
        self.local_work_modulus = None
        self.global_work_modulus = None
        self.packed_shape = None

        self.debug_plane_fill = False
        self.debug_plane_values = False
        self.debug_kernel_indices = False
        self.debug_kernel_launch = False

        super(KernelBuilder, self).__init__(index_name, reference_array_shape)

    def visit_IterationSpace(self, node):
        node = self.generic_visit(node)

        spaces = node.space.spaces
        num_spaces = len(spaces)

        total_work_dims, total_strides, total_lows = [], [], []
        self.packed_shape = [0 for _ in spaces[0].low]
        tiler = OclTiler(self.reference_array_shape, node, force_local_work_size=self.local_work_size)

        for space in spaces:
            lows = tuple(self.make_low(low, dim) for dim, low in enumerate(space.low))
            highs = tuple(self.make_high(high, dim) for dim, high in enumerate(space.high))
            strides = space.stride
            work_dims = []

            for dim, (low, high, stride) in reversed(list(enumerate(zip(lows, highs, strides)))):
                dim_size = (high - low + stride - 1) / stride
                work_dims.append(dim_size)
                self.packed_shape[dim] += dim_size

            total_work_dims.append(tuple(work_dims))
            total_strides.append(strides)
            total_lows.append(lows)

        self.global_work_size = total_work_dims[0]
        # self.global_work_size_1d = reduce(operator.mul, self.global_work_size)
        self.global_work_size = reduce(operator.mul, self.global_work_size)

        self.local_work_size = self.settings.force_local_work_size if self.settings.force_local_work_size is not None \
            else LocalSizeComputer(self.packed_shape, self.device).compute_local_size_bulky()
        self.local_work_size = reduce(operator.mul, self.local_work_size)

        # self.local_work_modulus = [x / num_spaces for x in self.local_work_size]
        # self.global_work_modulus = [x / num_spaces for x in self.global_work_size]

        # get_global_id(0)
        parts = [Assign(SymbolRef("global_id", ctypes.c_ulong()),
                        FunctionCall(SymbolRef("get_global_id"), [Constant(0)]))]
        # initialize index variables
        parts.extend(
            SymbolRef("{}_{}".format(self.index_name, dim), ctypes.c_ulong())
            for dim in range(len(self.reference_array_shape)))

        # calculate each index inline
        for space in range(len(node.space.spaces)):
            # indices = self.build_index_variables(SymbolRef("global_id"),
            #                                    shape=Vector(highs) - Vector(lows),
            #                                    multipliers=total_strides[space],
            #                                    offsets=total_lows[space])
            indices = tiler.global_index_to_coordinate_expressions(SymbolRef("global_id"),
                                                                   iteration_space_index=space)
            for dim in range(len(self.reference_array_shape)):
                parts.append(Assign(SymbolRef("{}_{}".format(self.index_name, dim)), indices[dim]))

            # for dim in range(tile.dim)
            new_body = [
                tiler.add_guards_if_necessary(statement)
                for statement in node.body
                ]
            node.body = new_body
            parts.extend(node.body)

        return MultiNode(parts)

    def packed_iteration_shape(self):
        self.packed_shape

    def number_type(self):
        return ctypes.c_double() if self.use_doubles else ctypes.c_float()

    def make_low(self, floor, dimension):
        return floor if floor >= 0 else self.reference_array_shape[dimension] + floor

    def make_high(self, ceiling, dimension):
        return ceiling if ceiling > 0 else self.reference_array_shape[dimension] + ceiling

    def get_additional_encode_funcs(self):
        return []
