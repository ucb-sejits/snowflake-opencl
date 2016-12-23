import ctypes
import operator
import pycl as cl
import ast
from ctree.c.macros import NULL
from ctree.c.nodes import Constant, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile, BinaryOp, ArrayRef, Add, Mod, Mul, Div, Lt, If, For, PostInc
from ctree.c.nodes import MultiNode, BitOrAssign
from ctree.jit import ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.types import get_ctype
# noinspection PyProtectedMember
from snowflake._compiler import find_names
from snowflake.compiler_utils import generate_encode_macro
from snowflake.stencil_compiler import Compiler, CCompiler
import math

from snowflake_opencl.nd_buffer import NDBuffer
from snowflake_opencl.ocl_tiler import OclTiler

__author__ = 'chick markley, seunghwan choi'

# TODO: move tiler computation inside of space expander
# TODO: fill planes


class PencilCompiler(Compiler):

    def __init__(self, context, device):
        super(PencilCompiler, self).__init__()
        self.context = context
        self.device = device

    BlockConverter = CCompiler.BlockConverter
    IndexOpToEncode = CCompiler.IndexOpToEncode

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
            self.compute_plane_info()

            self.global_work_size = None
            self.global_work_size_1d = None
            self.local_work_modulus = None

            super(PencilCompiler.PencilKernelBuilder, self).__init__(index_name, reference_array_shape)

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

            log_of_edge = int(math.log(max_size_per_dim - (self.ghost_size[0] * 2), 2))
            tile_edge = int(math.pow(2, log_of_edge)) + (self.ghost_size[0] * 2)

            self.plane_size = (tile_edge, tile_edge)
            self.plane_size_1d = reduce(operator.mul, self.plane_size)

            self.local_work_size = (tile_edge - (self.ghost_size[0] * 2), tile_edge - (self.ghost_size[0] * 2))
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
            self.global_work_size_1d = reduce(operator.mul, self.ghost_size)
            self.local_work_modulus = [x / num_spaces for x in self.local_work_size]
            self.global_work_modulus = [x / num_spaces for x in self.global_work_size]

            memory_declarations = self.get_local_memory_declarations()

            # get_global_id(0)
            parts = [
                memory_declarations,
                Assign(SymbolRef("tile_id_1", ctypes.c_ulong()), FunctionCall(SymbolRef("get_group_id"), [Constant(0)])),
                Assign(SymbolRef("tile_id_2", ctypes.c_ulong()), FunctionCall(SymbolRef("get_group_id"), [Constant(1)])),
                Assign(SymbolRef("packed_global_id_1", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
                Assign(SymbolRef("packed_global_id_2", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(1)])),
                Assign(SymbolRef("packed_local_id_1", ctypes.c_ulong()), FunctionCall(SymbolRef("get_local_id"), [Constant(0)])),
                Assign(SymbolRef("packed_local_id_2", ctypes.c_ulong()), FunctionCall(SymbolRef("get_local_id"), [Constant(1)])),
                Assign(
                    SymbolRef("thread_id", ctypes.c_ulong()),
                    Add(
                        Mul(SymbolRef("packed_local_id_1"), Constant(self.local_work_size[0])),
                        SymbolRef("packed_local_id_2")
                    )
                ),
                Assign(SymbolRef("group_id", ctypes.c_ulong()),FunctionCall(SymbolRef("get_group_id"), [Constant(0)])),
            ]

            # initialize index variables
            parts.extend(
                SymbolRef("{}_{}".format(self.index_name, dim), ctypes.c_ulong())
                for dim in range(len(self.reference_array_shape)))

            parts.extend(
                SymbolRef("{}_{}".format("local_" + self.index_name, dim), ctypes.c_ulong())
                for dim in range(len(self.reference_array_shape)))

            # construct offset arrays for each iterations space

            arrays = []
            for dim in range(0, 3):
                offsets = []
                strides = []
                for space in node.space.spaces:
                    offsets.append(space.low[dim])
                    strides.append(space.stride[dim])

                local_index = "local_index_{}".format(dim)
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
                    local_dim_mod = Mod(SymbolRef(packed_local_index), Constant(self.local_work_modulus[dim-1]))
                    global_dim_mod = Mod(SymbolRef(packed_global_index), Constant(self.local_work_modulus[dim-1]))

                    arrays.append(
                        Assign(
                            SymbolRef(local_index),
                            Add(
                                Mul(
                                    Mod(SymbolRef(packed_local_index), Constant(self.local_work_modulus[dim-1])),
                                    ArrayRef(SymbolRef(dim_strides), local_dim_mod)
                                ),
                                ArrayRef(SymbolRef(dim_offsets), local_dim_mod)
                            )
                        )
                    )
                    arrays.append(
                        Assign(
                            SymbolRef("index_{}".format(dim)),
                            Add(
                                Mul(
                                    Mod(SymbolRef(packed_global_index), Constant(self.global_work_modulus[dim-1])),
                                    ArrayRef(SymbolRef(dim_strides), global_dim_mod)
                                ),
                                ArrayRef(SymbolRef(dim_offsets), global_dim_mod)
                            )
                        )
                    )

            parts.extend(arrays)

            parts.append(StringTemplate("    //"))
            parts.append(StringTemplate("    // Fill local memory planes"))
            parts.append(StringTemplate("    //"))
            parts.extend(self.fill_plane2("plane_0", Constant(0)))
            parts.extend(self.fill_plane2("plane_1", Constant(1)))

            #
            # Do the pencil iteration
            for_body = [
                    self.fill_plane2("plane_2", Add(SymbolRef("index_0"), Constant(self.ghost_size[0]))),
                    StringTemplate('''barrier(CLK_LOCAL_MEM_FENCE);'''),
            ]
            for_body.extend(node.body)
            for_body.extend([
                Assign(SymbolRef("temp_plane"), SymbolRef("plane_0")),
                Assign(SymbolRef("plane_0"), SymbolRef("plane_1")),
                Assign(SymbolRef("plane_1"), SymbolRef("plane_2")),
                Assign(SymbolRef("plane_2"), SymbolRef("temp_plane")),
            ])
            pencil_block = For(
                init=Assign(SymbolRef("index_0"), Constant(self.ghost_size[0])),
                test=Lt(SymbolRef("index_0"), Constant(self.global_work_size[0])),
                incr=PostInc(SymbolRef("index_0")),
                body=for_body
            )

            parts.append(pencil_block)

            # parts.extend(node.body)

            return MultiNode(parts)

        def changingIndexofOut(self, binaryOp):
            offsetleft, offsetright = self.local_to_global_index()
            offsetleft._force_parentheses = True
            offsetright._force_parentheses = True
            binaryOp.right.args = [Add(offsetleft, SymbolRef(name="local_index_0")), Add(offsetright, SymbolRef(name="local_index_1"))]

        def changingMeshtoLocal(self, object, encodeFunc=None):
            if isinstance(object, BinaryOp):
                self.changingMeshtoLocal(object.left, encodeFunc)
                self.changingMeshtoLocal(object.right, encodeFunc)
            if isinstance(object, SymbolRef):
                if object.name == 'mesh':
                    object.name = 'localmem'
                if object.name == 'index_0' or object.name == 'index_1':
                    object.name = 'local_' + object.name

            if isinstance(object, FunctionCall):
                object.func = encodeFunc
                for x in object.args:
                    if isinstance(x, BinaryOp):
                        self.changingMeshtoLocal(x, encodeFunc)

        def fill_plane2(self, plane_name, index_0_expression):
            final = []
            local_size = self.local_work_size_1d
            copying_size = self.plane_size_1d

            index = 0
            while index < copying_size:
                local_location = Add(SymbolRef(name="thread_id"), Constant(index))
                local_location._force_parentheses = True
                left = ArrayRef(SymbolRef(name=plane_name), local_location)

                # arguments = self.local_to_global_index()
                sidearg0 = Div(local_location, Constant(self.plane_size[0]))
                sidearg0._force_parentheses = True
                sidearg1 = Mod(local_location, Constant(self.plane_size[1]))
                sidearg1._force_parentheses = True

                encode_arg0 = Add(
                    Add(
                        Mul(SymbolRef("tile_id_1"), Constant(self.local_work_size[0])),
                        Constant(self.ghost_size[1])), sidearg0)
                encode_arg1 = Add(
                    Add(
                        Mul(SymbolRef("tile_id_2"), Constant(self.local_work_size[1])),
                        Constant(self.ghost_size[2])), sidearg1)
                encode = FunctionCall(
                    func=SymbolRef("encode" + "_".join([str(x) for x in self.reference_array_shape])),
                    args=[index_0_expression, encode_arg0, encode_arg1]
                )

                right = ArrayRef(SymbolRef(name=self.stencil.primary_mesh), encode)

                if index + local_size > copying_size:
                    final.append(
                        If(
                            cond=Lt(Add(SymbolRef("thread_id"), Constant(index)), Constant(copying_size)),
                            then=Assign(left, right)
                        )
                    )
                else:
                    final.append(Assign(left, right))
                index += local_size
            return final

        def local_to_global_index(self):
            gid_x = self.packed_shape[0] / self.local_work_size[0]
            gid_y = self.packed_shape[1] / self.local_work_size[1]
            group0 = Div(SymbolRef('group_id'), Constant(gid_x))
            group0._force_parentheses = True
            group1 = Mod(SymbolRef('group_id'), Constant(gid_y))
            group1._force_parentheses = True
            offset0 = Mul(group0, Constant(self.local_work_size[0]))
            offset1 = Mul(group1, Constant(self.local_work_size[1]))
            return offset0, offset1


    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def __init__(self, context, global_work_size, local_work_size, kernels):
            self.context = context
            self.gws = global_work_size
            self.lws = local_work_size
            self.kernels = kernels
            self._c_function = None
            self.entry_point_name = None
            super(PencilCompiler.ConcreteSpecializedKernel, self).__init__()

        def finalize(self, entry_point_name, project_node, entry_point_typesig):
            self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
            self.entry_point_name = entry_point_name
            return self

        def __call__(self, *args, **kwargs):
            queue = cl.clCreateCommandQueue(self.context)
            true_args = [queue] + self.kernels + [arg.buffer if isinstance(arg, NDBuffer) else arg for arg in args]
            # this returns None instead of an int...
            return self._c_function(*true_args)

    # noinspection PyAbstractClass
    class LazySpecializedKernel(CCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, original=None, names=None, target_names=('out',), index_name='index',
                     _hash=None, context=None, device=None, local=False, loop=1):

            self.__hash = _hash if _hash is not None else hash(py_ast)
            self.names = names
            self.target_names = target_names
            self.index_name = index_name

            super(PencilCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )

            self.snowflake_ast = original
            self.parent_cls = PencilCompiler
            self.context = context
            self.device = device
            self.global_work_size = 0
            self.local_work_size = 0
            self.local = local
            self.loop = loop

        def insert_indexing_debugging_printfs(self, shape, name_index=None):
            format_string = 'wgid %03d gid %04d'
            argument_string = 'get_group_id(0), global_id,'
            # noinspection PyProtectedMember
            encode_string = 'encode'+CCompiler._shape_to_str(shape)

            index_variables = ["{}_{}".format(self.index_name, dim) for dim in range(len(shape))]

            format_string += " index (" + ", ".join("%d".format(var) for var in index_variables) + ") "
            argument_string += " " + ", ".join("{}".format(var) for var in index_variables)

            format_string += " " + encode_string + "(" + ", ".join("{}".format(var) for var in index_variables) + ") %d"
            argument_string += ", " + encode_string + "(" + ", ".join("{}".format(var) for var in index_variables) + ")"

            # TODO: Fix this, seems like this.names is a set and does not like the index reference
            # if name_index is not None:
            #     format_string += " " + self.names[name_index] + "[" + encode_string + \
            #                      "(" + ", ".join("{}".format(var) for var in index_variables) + ")] %f"
            #     argument_string += ", " + self.names[name_index] + "[" + encode_string + \
            #                        "(" + ", ".join("{}".format(var) for var in index_variables) + ")]"

            return StringTemplate('printf("{}\\n", {});'.format(format_string, argument_string))

        def transform(self, tree, program_config):
            """
            The tree is a snowflake stencil group which makes it one or more
            Stencil Operations.  This
            :param tree:
            :param program_config:
            :return:
            """

            subconfig, tuning_config = program_config
            name_shape_map = {name: arg.shape for name, arg in subconfig.items()}
            shapes = set(name_shape_map.values())
            self.parent_cls.IndexOpToEncode(name_shape_map).visit(tree)
            c_tree = PyBasicConversions().visit(tree)

            encode_funcs = []
            for shape in shapes:
                # noinspection PyProtectedMember
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape), shape))

            includes = [StringTemplate("#pragma OPENCL EXTENSION cl_khr_fp64 : enable")]  # should add this to ctree

            ocl_files, kernels = [], []
            control = [Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0))]
            error_code = SymbolRef("error_code")
            gws_arrays, lws_arrays = {}, {}

            # build a bunch of kernels
            #
            for kernel_number, (target, i_space, stencil_node) in enumerate(
                    zip(self.target_names, c_tree.body, self.snowflake_ast.body)):

                kernel_builder = self.parent_cls.PencilKernelBuilder(
                    self.index_name,
                    subconfig[target].shape,
                    stencil_node,
                    self.device,
                )

                kernel_body = kernel_builder.visit(i_space)
                kernel_body = self.parent_cls.BlockConverter().visit(kernel_body)  # changes node to MultiNode

                local_reference_shape = kernel_builder.plane_size
                new_encode_func = generate_encode_macro(
                    'encode' + CCompiler._shape_to_str(local_reference_shape),local_reference_shape
                )
                if new_encode_func not in encode_funcs:
                    encode_funcs.append(new_encode_func)  # local

                local_work_size_1d = kernel_builder.local_work_size_1d
                print(
                    "local_work_size {} local_work_size_1d {}".format(
                        kernel_builder.local_work_size, local_work_size_1d
                    )
                )

                # Uncomment the following line to put some printf showing index values at runtime
                # kernel_body.body.append(self.insert_indexing_debugging_printfs(shape))
                # Or uncomment this to be able to print data from one of the buffer names, by specifying name index
                # kernel_body.body.append(self.insert_indexing_debugging_printfs(shape, name_index=0))

                kernel_params = [
                    SymbolRef(name=arg_name, sym_type=get_ctype(
                         arg if not isinstance(arg, NDBuffer) else arg.ary.ravel()  # hack
                     ), _global=True) for arg_name, arg in subconfig.items()
                    ]
                kernel_func = FunctionDecl(
                    name=SymbolRef("kernel_{}".format(kernel_number)),
                    params=kernel_params,
                    defn=[kernel_body]
                )

                kernel_func.set_kernel()
                kernels.append(kernel_func)
                ocl_files.append(OclFile(name=kernel_func.name.name, body=includes + encode_funcs + [kernel_func]))

                gws = kernel_builder.global_work_size_1d

                # declare new global and local work size arrays if necessary
                # if gws % local_work_size_1d > 0:
                #     expanded_packed = [
                #         (int(kernel_builder.packed_iteration_shape[dim] / kernel_builder.local_work_size[dim]) + 1) * kernel_builder.local_work_size[dim]
                #         for dim in range(len(kernel_builder.local_work_size))
                #     ]
                #     gws = reduce(operator.mul, expanded_packed)
                if gws not in gws_arrays:
                    control.append(
                        ArrayDef(SymbolRef("global_%d " % gws, ctypes.c_ulong()), 1, Array(body=[Constant(gws)])))
                    gws_arrays[gws] = SymbolRef("global_%d" % gws)

                if local_work_size_1d not in lws_arrays:
                    control.append(
                        ArrayDef(
                            SymbolRef("local_%d " % local_work_size_1d, ctypes.c_ulong()),
                            1,
                            Array(body=[Constant(local_work_size_1d)])
                        )
                    )
                    lws_arrays[local_work_size_1d] = SymbolRef("local_%d" % local_work_size_1d)

                # clSetKernelArg
                for arg_num, arg in enumerate(kernel_func.params):
                    set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                           [SymbolRef(kernel_func.name.name),
                                            Constant(arg_num),
                                            Constant(ctypes.sizeof(cl.cl_mem)),
                                            Ref(SymbolRef(arg.name))])
                    control.append(BitOrAssign(error_code, set_arg))

                # clEnqueueNDRangeKernel
                enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
                                   SymbolRef("queue"), SymbolRef(kernel_func.name), Constant(1), NULL(),
                                   gws_arrays[gws], lws_arrays[local_work_size_1d], Constant(0), NULL(), NULL()
                               ])
                control.append(BitOrAssign(error_code, enqueue_call))
                control.append(StringTemplate("""clFinish(queue);"""))

            #control.append(StringTemplate("""clFinish(queue);"""))
            control.append(StringTemplate("if (error_code != 0) printf(\"error code %d\\n\", error_code);"))
            control.append(Return(SymbolRef("error_code")))
            # should do bit or assign for error code

            control_params = [SymbolRef("queue", cl.cl_command_queue())]
            for kernel_func in kernels:
                control_params.append(SymbolRef(kernel_func.name.name, cl.cl_kernel()))
            control_params.extend([
                    SymbolRef(name=arg_name, sym_type=get_ctype(
                         arg) if not isinstance(arg, NDBuffer) else cl.cl_mem()  # hack
                     ) for arg_name, arg in subconfig.items()
                   ])

            control = FunctionDecl(return_type=ctypes.c_int32(), name="control", params=control_params, defn=control)
            ocl_include = StringTemplate("""
                            #include <stdio.h>
                            #include <time.h>
                            #ifdef __APPLE__
                            #include <OpenCL/opencl.h>
                            #else
                            #include <CL/cl.h>
                            #endif
                            """)

            c_file = CFile(name="control", body=[ocl_include, control], config_target='opencl')
            print(c_file)
            for f in ocl_files:
                print("{}".format(f.codegen()))
            return [c_file] + ocl_files

        def finalize(self, transform_result, program_config):

            project = Project(files=transform_result)
            kernels = []
            for i, f in enumerate(transform_result[1:]):
                kernels.append(cl.clCreateProgramWithSource(self.context, f.codegen()).build()["kernel_%d" % i])
            fn = PencilCompiler.ConcreteSpecializedKernel(
                self.context, self.global_work_size, self.local_work_size, kernels)
            func_types = [cl.cl_command_queue] + [cl.cl_kernel for _ in range(len(kernels))] + [
                        cl.cl_mem if isinstance(arg, NDBuffer) else type(arg)
                        for arg in program_config.args_subconfig.values()
                    ]
            return fn.finalize(
                entry_point_name='control',  # not used
                project_node=project,
                entry_point_typesig=ctypes.CFUNCTYPE(
                    None, *func_types
                )
            )

    def _post_process(self, original, compiled, index_name, local=False, loop=1):
        return self.LazySpecializedKernel(
            py_ast=compiled,
            original=original,
            names=find_names(original),
            index_name=index_name,
            target_names=[stencil.primary_mesh for stencil in original.body if hasattr(stencil, "primary_mesh")],
            _hash=hash(original),
            context=self.context,
            device=self.device,
            local=local,
            loop=loop
        )
