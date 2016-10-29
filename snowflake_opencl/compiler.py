import ctypes
import operator
import pycl as cl
from ctree.c.macros import NULL
from ctree.c.nodes import Constant, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile, BinaryOp, ArrayRef, Add, Mod, Mul, Div
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

from snowflake_opencl.ocl_tiler import OclTiler

__author__ = 'dorthy luu'


class NDBuffer(object):
    def __init__(self, queue, ary, blocking=True):
        self.ary = ary
        self.shape = ary.shape
        self.dtype = ary.dtype
        self.ndim = ary.ndim
        self.queue = queue
        self.buffer, evt = cl.buffer_from_ndarray(queue, ary)
        if blocking:
            evt.wait()

class OpenCLCompiler(Compiler):

    def __init__(self, context, local=False,loop=1):
        super(OpenCLCompiler, self).__init__()
        self.context = context
        self.local = local
        self.loop = loop

    BlockConverter = CCompiler.BlockConverter
    IndexOpToEncode = CCompiler.IndexOpToEncode

    class TiledIterationSpaceExpander(CCompiler.IterationSpaceExpander):
        def __init__(self, index_name, reference_array_shape, tiler, local=False):
            self.packed_shape = tiler.packed_iteration_shape
            self.local_work_size = tiler.local_work_size
            super(OpenCLCompiler.TiledIterationSpaceExpander, self).__init__(index_name, reference_array_shape)

        def make_low(self, floor, dimension):
            return floor if floor >= 0 else self.reference_array_shape[dimension] + floor

        def make_high(self, ceiling, dimension):
            return ceiling if ceiling > 0 else self.reference_array_shape[dimension] + ceiling


        def visit_IterationSpace(self, node):
            node = self.generic_visit(node)

            total_work_dims, total_strides, total_lows = [], [], []
            tiler = OclTiler(self.reference_array_shape, node, force_local_work_size=self.packed_shape)
            local_tiler = OclTiler(self.local_work_size, node, force_local_work_size=self.local_work_size)
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

            # get_global_id(0)
            parts = [Assign(SymbolRef("global_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_global_id"), [Constant(0)])),
                     Assign(SymbolRef("local_id", ctypes.c_ulong()), FunctionCall(SymbolRef("get_local_id"), [Constant(0)])),
                     Assign(SymbolRef("group_id", ctypes.c_ulong()),FunctionCall(SymbolRef("get_group_id"), [Constant(0)])),
                     StringTemplate("""__local float localmem[25];""")]
            #get local_id(0)
            # initialize index variables
            parts.extend(
                SymbolRef("{}_{}".format(self.index_name, dim), ctypes.c_ulong())
                for dim in range(len(self.reference_array_shape)))

            parts.extend(
                SymbolRef("{}_{}".format("local_" + self.index_name, dim), ctypes.c_ulong())
                for dim in range(len(self.reference_array_shape)))
            # parts.extend(
            #     SymbolRef("*localmem", ctypes.c_float(), _local=True)
            # )

            # calculate each index inline
            for space in range(len(node.space.spaces)):
                # indices = self.build_index_variables(SymbolRef("global_id"),
                #                                    shape=Vector(highs) - Vector(lows),
                #                                    multipliers=total_strides[space],
                #                                    offsets=total_lows[space])
                indices = tiler.global_index_to_coordinate_expressions(SymbolRef("global_id"),
                                                                       iteration_space_index=space)
                local_indices = local_tiler.global_index_to_coordinate_expressions(SymbolRef("local_id"), iteration_space_index=space)

                for dim in range(len(self.reference_array_shape)):
                    parts.append(Assign(SymbolRef("{}_{}".format(self.index_name, dim)), indices[dim]))
                for dim in range(len(self.reference_array_shape)):
                    parts.append(Assign(SymbolRef("{}_{}".format("local_" + self.index_name, dim)), local_indices[dim]))

                # for dim in range(tile.dim)
                new_body = [
                    tiler.add_guards_if_necessary(statement)
                    for statement in node.body
                    ]
                node.body = new_body
                parts.extend(self.local_to_global_copy())
                local_reference_array_shape = self.calculate_local_reference_array_shape()
                parts.append((StringTemplate('''barrier(CLK_LOCAL_MEM_FENCE);''')))
                encodeFunc = SymbolRef("encode" + str(local_reference_array_shape[0]) + "_" + str(local_reference_array_shape[0]))
                self.changingMeshtoLocal(node.body[0].right, encodeFunc)
                self.changingIndexofOut(node.body[0].left)
                parts.extend(node.body)

                #parts.append(StringTemplate('wait();'))
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

        def calculate_local_reference_array_shape(self):
            ghost_size = self.reference_array_shape[0] - self.packed_shape[0]
            local_reference_array_shape = []
            for x in self.local_work_size:
                local_reference_array_shape.append(x + ghost_size)
            return local_reference_array_shape

        def local_to_global_copy(self):
            final = []
            localSize = reduce(operator.mul, self.local_work_size)
            local_reference_array_shape = self.calculate_local_reference_array_shape()
            copyingSize = reduce(operator.mul, local_reference_array_shape)
            index = 0
            while index < copyingSize:
                local_location = Add(SymbolRef(name="local_id"), Constant(index))
                local_location._force_parentheses = True
                left = ArrayRef(SymbolRef(name="localmem"), local_location)
                arguments = self.local_to_global_index()
                sidearg0 = Div(local_location, Constant(local_reference_array_shape[0]))
                sidearg0._force_parentheses = True
                sidearg1 = Mod(local_location, Constant(local_reference_array_shape[1]))
                sidearg1._force_parentheses = True
                encode_arg0 = Add(arguments[0],sidearg0 )
                encode_arg1 = Add(arguments[1], sidearg1)
                encode = FunctionCall(func=SymbolRef("encode" + str(self.reference_array_shape[0]) + "_" + str(self.reference_array_shape[0])), args=[encode_arg0, encode_arg1])
                right = ArrayRef(SymbolRef(name="mesh"), encode)
                if index + localSize > copyingSize:
                    ifstatement = 'if (local_id + ' + str(index) + ' < ' + str(copyingSize) + ') {'
                    ifstatement = StringTemplate(ifstatement)
                    final.append(ifstatement)
                    final.append((Assign(left, right)))
                    final.append(StringTemplate('}'))
                else:
                    final.append(Assign(left, right))
                index+= localSize
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
            return (offset0, offset1)


    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def __init__(self, context, global_work_size, local_work_size, kernels):
            self.context = context
            self.gws = global_work_size
            self.lws = local_work_size
            self.kernels = kernels
            self._c_function = None
            self.entry_point_name = None
            super(OpenCLCompiler.ConcreteSpecializedKernel, self).__init__()

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
                     _hash=None, context=None, local=False, loop=1):

            self.__hash = _hash if _hash is not None else hash(py_ast)
            self.names = names
            self.target_names = target_names
            self.index_name = index_name

            super(OpenCLCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )

            self.snowflake_ast = original
            self.parent_cls = OpenCLCompiler
            self.context = context
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
            The tree is based on a snowflake stencil group which makes it one or more
            Stencil Operations.  This
            :param tree:
            :param program_config:
            :return:
            """
            subconfig, tuning_config = program_config
            name_shape_map = {name: arg.shape for name, arg in subconfig.items()}
            shapes = set(name_shape_map.values())
            local_work_size = (5, 5)
            self.parent_cls.IndexOpToEncode(name_shape_map).visit(tree)
            c_tree = PyBasicConversions().visit(tree)

            encode_funcs = []
            for shape in shapes:
                # noinspection PyProtectedMember
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape), shape))
                encode_funcs.append(generate_encode_macro('encode' + CCompiler._shape_to_str(local_work_size), local_work_size))

            includes = [StringTemplate("#pragma OPENCL EXTENSION cl_khr_fp64 : enable")]  # should add this to ctree

            ocl_files, kernels = [], []
            control = [Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0))]
            error_code = SymbolRef("error_code")
            gws_arrays, lws_arrays = {}, {}

            # build a bunch of kernels
            #
            for i, (target, i_space, stencil_node) in enumerate(
                    zip(self.target_names, c_tree.body, self.snowflake_ast.body)):

                shape = subconfig[target].shape

                tiler = OclTiler(shape, i_space, force_local_work_size=(3,3))
                packed_iteration_shape = tiler.packed_iteration_shape

                gws = reduce(operator.mul, packed_iteration_shape)

                local_work_size = tiler.local_work_size
                # local_work_size = LocalSizeComputer(packed_iteration_shape).compute_local_size_bulky()

                local_work_size_1d = reduce(operator.mul, local_work_size)
                print("local_work_size {} local_work_size_1d {}".format(local_work_size, local_work_size_1d))

                sub = self.parent_cls.TiledIterationSpaceExpander(
                    self.index_name,
                    shape,
                    tiler).visit(i_space)
                sub = self.parent_cls.BlockConverter().visit(sub)  # changes node to MultiNode

                # Uncomment the following line to put some printf showing index values at runtime
                # sub.body.append(self.insert_indexing_debugging_printfs(shape))
                # Or uncomment this to be able to print data from one of the buffer names, by specifying name index
                # sub.body.append(self.insert_indexing_debugging_printfs(shape, name_index=0))

                kernel_params = [
                    SymbolRef(name=arg_name, sym_type=get_ctype(
                         arg if not isinstance(arg, NDBuffer) else arg.ary.ravel()  # hack
                     ), _global=True) for arg_name, arg in subconfig.items()
                   ]
                kernel_func = FunctionDecl(name=SymbolRef("kernel_%d" % i), params=kernel_params, defn=[sub])
                kernel_func.set_kernel()
                kernels.append(kernel_func)
                ocl_files.append(OclFile(name=kernel_func.name.name, body=includes + encode_funcs + [kernel_func]))

                # declare new global and local work size arrays if necessary
                if gws % local_work_size_1d > 0:
                    expanded_packed = [
                        (int(packed_iteration_shape[dim] / local_work_size[dim]) + 1) * local_work_size[dim]
                        for dim in range(len(local_work_size))
                    ]
                    gws = reduce(operator.mul, expanded_packed)
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
            fn = OpenCLCompiler.ConcreteSpecializedKernel(
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
            local=local,
            loop=loop
        )
