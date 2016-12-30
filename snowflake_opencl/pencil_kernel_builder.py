import ctypes
import operator

from ctree.c.nodes import Constant, SymbolRef, Assign, FunctionCall, \
    ArrayRef, Add, Mod, Mul, Div, Lt, If, For, PostInc, LtE
from ctree.c.nodes import MultiNode
from ctree.templates.nodes import StringTemplate

# noinspection PyProtectedMember
from snowflake.stencil_compiler import CCompiler
import math

from snowflake_opencl.primary_mesh_to_plane_transformer import PrimaryMeshToPlaneTransformer

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


class PencilKernelBuilder(CCompiler.IterationSpaceExpander):
    def __init__(self, index_name, reference_array_shape, stencil, device):
        self.stencil = stencil
        self.stencil_node = stencil.op_tree
        self.ghost_size = tuple((x - 1)/2 for x in self.stencil_node.weights.shape)
        self.device = device
        self.use_doubles = "cl_khr_fp64" in self.device.extensions

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

        self.debug_plane_fill = False
        self.debug_plane_values = False
        self.debug_kernel_indices = False
        self.debug_kernel_launch = False

        super(PencilKernelBuilder, self).__init__(index_name, reference_array_shape)

    def number_size(self):
        return 8 if self.use_doubles else 4

    def number_type(self):
        return "double" if self.use_doubles else "float"

    def make_low(self, floor, dimension):
        return floor if floor >= 0 else self.reference_array_shape[dimension] + floor

    def make_high(self, ceiling, dimension):
        return ceiling if ceiling > 0 else self.reference_array_shape[dimension] + ceiling

    def compute_plane_info(self):
        """
        assumes for the present that the problem is symmetric

        compute the following:
            how many numbers can local memory hold
            divide that by the number of planes needed
            take the square root of that to figure the edge size of the tile
            for now take the largest power of 2 that so things fit nice (our problem is almost always sized
            as a power of 2
        """
        max_reals_in_localmem = self.device.local_mem_size / self.number_size()
        max_real_nums_per_plane = max_reals_in_localmem / self.planes
        max_size_per_dim = math.sqrt(max_real_nums_per_plane)

        max_local_work_size_per_dim = int(math.sqrt(self.device.max_work_group_size))

        log_of_edge = int(math.log(max_size_per_dim - (self.ghost_size[0] * 2), 2))
        log_of_edge = min(log_of_edge, int(math.log(max_local_work_size_per_dim, 2)))
        log_of_edge = min(log_of_edge, int(math.log(self.global_work_size[1], 2)))
        lws_1 = int(math.pow(2, log_of_edge))
        lws_2 = lws_1
        if lws_1 * lws_2 < self.device.max_work_group_size and lws_1 * (lws_2 * 2) <= self.device.max_work_group_size:
            lws_2 *= 2

        # tile_edge = 4  #TODO: remove this

        self.plane_size = (lws_1 + (self.ghost_size[0] * 2), lws_2 + (self.ghost_size[1] * 2))
        self.plane_size_1d = reduce(operator.mul, self.plane_size)

        self.local_work_size = (lws_1, lws_2)
        self.local_work_size_1d = reduce(operator.mul, self.local_work_size)

    def get_local_memory_declarations(self):
        """
        created a local memory buffer for each plane
        and create a pointer pointing to each plane
        :return: a StringTemplate with the opencl code for the planes
        """
        buffers = [
            "__local {} local_buf_{}[{}];".format(self.number_type(), n, self.plane_size_1d)
            for n in range(self.planes)
            ]
        pointers = [
            "__local {}* plane_{} = local_buf_{};".format(self.number_type(), n, n)
            for n in range(self.planes)
            ]
        pointers += ["__local {}* temp_plane;".format(self.number_type())]
        string = '\n'.join(buffers + pointers)

        return StringTemplate(string)

    def visit_IterationSpace(self, node):
        node = self.generic_visit(node)

        total_work_dims, total_strides, total_lows = [], [], []

        spaces = node.space.spaces
        num_spaces = len(spaces)
        for space in node.space.spaces:
            lows = tuple(self.make_low(low, dim) for dim, low in enumerate(space.low))
            highs = tuple(self.make_high(high, dim) for dim, high in enumerate(space.high))
            strides = space.stride
            work_dims = []

            for dim, (low, high, stride) in reversed(list(enumerate(zip(lows, highs, strides)))):
                work_dims.append((high - low + stride - 1) / stride)

            total_work_dims.append(tuple(work_dims))
            total_strides.append(strides)
            total_lows.append(lows)

        self.global_work_size = total_work_dims[0][1:]
        self.global_work_size_1d = reduce(operator.mul, self.global_work_size)

        self.compute_plane_info()

        self.local_work_modulus = [x / num_spaces for x in self.local_work_size]
        self.global_work_modulus = [x / num_spaces for x in self.global_work_size]

        memory_declarations = self.get_local_memory_declarations()

        # get_global_id(0)
        parts = [
            memory_declarations,
            Assign(
                SymbolRef("tile_id_1", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_group_id"), [Constant(0)])),
            Assign(
                SymbolRef("tile_id_2", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_group_id"), [Constant(1)])),
            Assign(
                SymbolRef("packed_global_id_1", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
            Assign(
                SymbolRef("packed_global_id_2", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_global_id"), [Constant(1)])),
            Assign(
                SymbolRef("packed_local_id_1", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_local_id"), [Constant(0)])),
            Assign(
                SymbolRef("packed_local_id_2", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_local_id"), [Constant(1)])),
            Assign(
                SymbolRef("thread_id", ctypes.c_ulong()),
                Add(
                    Mul(SymbolRef("packed_local_id_1"), Constant(self.local_work_size[1])),
                    SymbolRef("packed_local_id_2")
                )
            ),
            Assign(
                SymbolRef("group_id_0", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_group_id"), [Constant(0)])),
            Assign(
                SymbolRef("group_id_1", ctypes.c_ulong()),
                FunctionCall(SymbolRef("get_group_id"), [Constant(1)])),
        ]

        # initialize index variables
        parts.extend(
            SymbolRef("{}_{}".format(self.index_name, dim), ctypes.c_ulong())
            for dim in range(len(self.reference_array_shape)))
        parts.extend(
            SymbolRef("local_{}_{}".format(self.index_name, dim), ctypes.c_ulong())
            for dim in range(len(self.reference_array_shape)))

        # construct offset arrays for each iterations space

        arrays = []
        for dim in range(0, 3):
            offsets = []
            strides = []
            for space in node.space.spaces:
                offsets.append(space.low[dim])
                strides.append(space.stride[dim])

            packed_local_index = "packed_local_id_{}".format(dim)
            packed_global_index = "packed_global_id_{}".format(dim)
            dim_offsets = "dim_{}_offsets".format(dim)
            dim_strides = "dim_{}_strides".format(dim)
            arrays.append(
                Assign(
                    StringTemplate("size_t {}[]".format(dim_offsets)),
                    StringTemplate('{' + ", ".join([str(x) for x in offsets]) + '}')
                )
            )
            arrays.append(
                Assign(
                    StringTemplate("size_t {}[]".format(dim_strides)),
                    StringTemplate('{' + ", ".join([str(x) for x in strides]) + '}')
                )
            )

            if dim > 0:
                space_index = Div(SymbolRef(packed_global_index), Constant(self.global_work_modulus[dim - 1]))
                space_index._force_parentheses = True
                modded_gobal_index = Mod(
                    SymbolRef(packed_global_index), Constant(self.global_work_modulus[dim - 1]))
                modded_gobal_index._force_parentheses = True
                modded_local_index = Mod(
                    SymbolRef(packed_local_index), Constant(self.global_work_modulus[dim - 1]))
                modded_local_index._force_parentheses = True

                arrays.append(
                    Assign(
                        SymbolRef("index_{}".format(dim)),
                        Add(
                            Mul(
                                modded_gobal_index,
                                ArrayRef(SymbolRef(dim_strides), space_index)
                            ),
                            ArrayRef(SymbolRef(dim_offsets), space_index)
                        )
                    )
                )

                arrays.append(
                    Assign(
                        SymbolRef("local_index_{}".format(dim)),
                        Add(
                            Mul(
                                modded_local_index,
                                ArrayRef(SymbolRef(dim_strides), space_index)
                            ),
                            ArrayRef(SymbolRef(dim_offsets), space_index)
                        )
                    )
                )

        def debug_show_plane(plane_number):
            if not self.debug_plane_values:
                return []

            elements = self.plane_size[0]
            return [
                StringTemplate(
                    'if(thread_id == 0) {{printf("{}\\plane_number", {});}}'.format(
                       "pfwg %4d %4d  %4d   " + ",".join(["%6.4f " for _ in range(elements)]),
                       "group_id_0, group_id_1, " + str(plane_number) + ", " +
                       ",".join(["plane_" + str(plane_number) + "[{}]".format(
                           (y * elements) + x_val) for x_val in range(elements)])
                    )
                )
                for y in range(elements)
            ]

        parts.extend(arrays)

        if self.debug_kernel_launch:
            parts.append(
                StringTemplate(
                    'if(thread_id == 0) {printf("launching kernel %3d%3d\\n", group_id_0, group_id_1);}'))

        parts.append(StringTemplate("    //"))
        parts.append(StringTemplate("    // Fill the first local memory planes"))
        parts.append(StringTemplate("    //"))
        parts.extend(self.fill_plane("plane_1", Constant(0)))
        parts.extend(debug_show_plane(1))
        parts.extend(self.fill_plane("plane_2", Constant(1)))
        parts.extend(debug_show_plane(2))
        parts.extend([
            StringTemplate('''barrier(CLK_LOCAL_MEM_FENCE);'''),
        ])

        for_body = []
        # for_body.append(
        #     StringTemplate("out[encode10_10_10(index_0, index_1, index_2)] = 12.34;")
        # )
        #
        # Do the pencil iteration
        body_transformer = PrimaryMeshToPlaneTransformer(self.stencil_node.name, self.plane_size)
        new_body = [body_transformer.visit(sub_node) for sub_node in node.body]

        for_body.extend([
            Assign(SymbolRef("temp_plane"), SymbolRef("plane_0")),
            Assign(SymbolRef("plane_0"), SymbolRef("plane_1")),
            Assign(SymbolRef("plane_1"), SymbolRef("plane_2")),
            Assign(SymbolRef("plane_2"), SymbolRef("temp_plane")),
        ])

        for_body.extend([
            self.fill_plane("plane_2", Add(SymbolRef("index_0"), Constant(self.ghost_size[0]))),
            StringTemplate('''barrier(CLK_LOCAL_MEM_FENCE);'''),
        ])
        for_body.extend(debug_show_plane(2))
        for_body.extend(new_body)
        # for_body.extend(node.body)
        for_body.extend([
            StringTemplate('''barrier(CLK_LOCAL_MEM_FENCE);'''),
        ])

        if self.debug_kernel_indices:
            for_body.append(
                StringTemplate(
                   'printf("{}\\n", {});'.format(
                       "group (%3d, %3d) thread %d packed_global_id (%5d, %5d) " +
                       "index (%4d, %4d, %4d) local_index (%4d, %4d, %4d) out %6.4f",
                       "group_id_0, group_id_1, thread_id, packed_global_id_1, packed_global_id_2" +
                       ", index_0, index_1, index_2" +
                       ", local_index_0, local_index_1, local_index_2" +
                       ", out[encode{}(index_0, index_1, index_2)]".format(
                           "_".join([str(n) for n in self.reference_array_shape])
                       )
                   )
                )
            )

        pencil_block = For(
            init=Assign(SymbolRef("index_0"), Constant(self.ghost_size[0])),
            test=LtE(SymbolRef("index_0"), Constant(self.global_work_size[0])),
            # test=LtE(SymbolRef("index_0"), Constant(self.ghost_size[0])),
            incr=PostInc(SymbolRef("index_0")),
            body=for_body
        )

        parts.append(pencil_block)

        return MultiNode(parts)

    def fill_plane(self, plane_name, index_0_expression):
        final = []
        local_size = self.local_work_size_1d
        copying_size = self.plane_size_1d

        index = 0
        while index < copying_size:
            local_location = Add(SymbolRef(name="thread_id"), Constant(index))
            local_location._force_parentheses = True
            left = ArrayRef(SymbolRef(name=plane_name), local_location)

            index_1 = Div(local_location, Constant(self.plane_size[1]))
            index_1._force_parentheses = True
            index_2 = Mod(local_location, Constant(self.plane_size[1]))
            index_2._force_parentheses = True

            # (thread_id + offset) % tile_id
            encode_arg0 = Add(
                    Mul(SymbolRef("tile_id_1"), Constant(self.local_work_size[0])), index_1)
            encode_arg0._force_parentheses = True

            encode_arg1 = Add(
                    Mul(SymbolRef("tile_id_2"), Constant(self.local_work_size[1])), index_2)
            encode_arg1._force_parentheses = True

            encode = FunctionCall(
                func=SymbolRef("encode" + "_".join([str(x) for x in self.reference_array_shape])),
                args=[index_0_expression, encode_arg0, encode_arg1]
            )

            right = ArrayRef(SymbolRef(name=self.stencil.op_tree.name), encode)

            def make_debug_printf():
                if not self.debug_plane_fill:
                    return StringTemplate("")

                return StringTemplate(
                    ' printf("plane fill {}\\n", {});'.format(
                       "%6d, %6d, %6d, " + plane_name + "[ %6d ] = %6.4f",
                       index_0_expression.codegen() + ", " + encode_arg0.codegen() + ", " + encode_arg1.codegen() +
                       ", " + local_location.codegen() + ",  " + right.codegen()
                    )
                )

            if index + local_size > copying_size:
                final.append(
                    If(
                        cond=Lt(Add(SymbolRef("thread_id"), Constant(index)), Constant(copying_size)),
                        then=[
                            Assign(left, right),
                            make_debug_printf()
                        ]
                    )
                )
            else:
                final.append(Assign(left, right))
                final.append(make_debug_printf())

            index += local_size
        return final
