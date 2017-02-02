import ctypes
import operator

import pycl as cl
import time
from ctree.c.macros import NULL
from ctree.c.nodes import Constant, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile, For, LtE, PostInc, Lt
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

from snowflake_opencl.kernel_builder import KernelBuilder
from snowflake_opencl.nd_buffer import NDBuffer
from snowflake_opencl.ocl_tiler import OclTiler

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


class OpenCLCompiler(Compiler):

    def __init__(self, context, device=None, settings=None):
        super(OpenCLCompiler, self).__init__()
        self.context = context
        self.device = device if device else cl.clGetDeviceIDs()[-1]
        self.settings = settings

    BlockConverter = CCompiler.BlockConverter
    IndexOpToEncode = CCompiler.IndexOpToEncode

    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def __init__(self, context, global_work_size, local_work_size, kernels, label="pencil"):
            self.context = context
            self.gws = global_work_size
            self.lws = local_work_size
            self.kernels = kernels
            self._c_function = None
            self.entry_point_name = None
            self.label = label
            super(OpenCLCompiler.ConcreteSpecializedKernel, self).__init__()

        def finalize(self, entry_point_name, project_node, entry_point_typesig):
            self._c_function = self._compile(entry_point_name, project_node, entry_point_typesig)
            self.entry_point_name = entry_point_name
            return self

        def __call__(self, *args, **kwargs):
            queue = cl.clCreateCommandQueue(self.context)
            true_args = [queue] + self.kernels + [arg.buffer if isinstance(arg, NDBuffer) else arg for arg in args]
            # this returns None instead of an int...

            start_time = time.time()
            result = self._c_function(*true_args)
            end_time = time.time()
            print("{:10.5f} {}".format((end_time - start_time), self.label))
            return result

    # noinspection PyAbstractClass
    class LazySpecializedKernel(CCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, original=None, names=None, target_names=('out',), index_name='index',
                     _hash=None, context=None, device=None, settings=None):

            self.__hash = _hash if _hash is not None else hash(py_ast)
            self.names = names
            self.target_names = target_names
            self.index_name = index_name
            self.settings = settings

            super(OpenCLCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )

            self.snowflake_ast = original
            self.parent_cls = OpenCLCompiler
            self.context = context
            self.device = device
            self.global_work_size = 0
            self.local_work_size = 0

        # noinspection PyUnusedLocal
        def insert_indexing_debugging_printfs(self, shape, name_index=None):
            format_string = 'wgid %03d gid1 %04d gid2 %04d'
            argument_string = 'get_group_id(0), packed_global_id_1, packed_global_id_2,'
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

        def build_regular_kernel(self):
            None

        # noinspection PyProtectedMember
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
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape), shape))
                # the second one here is for indexing the local memory planes
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape[1:]), shape[1:]))

            includes = []
            if "cl_khr_fp64" in self.device.extensions:
                includes.append(StringTemplate("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"))

            ocl_files, kernels = [], []
            control = [Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0))]
            error_code = SymbolRef("error_code")
            gws_arrays, lws_arrays = {}, {}

            # build a bunch of kernels
            #
            for kernel_number, (target, i_space, stencil_node) in enumerate(
                    zip(self.target_names, c_tree.body, self.snowflake_ast.body)):

                kernel_builder = KernelBuilder(
                    self.index_name,
                    subconfig[target].shape,
                    stencil_node,
                    self.context,
                    self.device,
                    self.settings
                )

                kernel_body = kernel_builder.visit(i_space)
                kernel_body = self.parent_cls.BlockConverter().visit(kernel_body)  # changes node to MultiNode

                for new_encode_func in kernel_builder.get_additional_encode_funcs():
                    if new_encode_func not in encode_funcs:
                        encode_funcs.append(new_encode_func)

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

                gws = kernel_builder.global_work_size
                lws = kernel_builder.local_work_size

                if gws not in gws_arrays:
                    if(isinstance(gws, tuple)):
                        control.append(
                            ArrayDef(
                                SymbolRef("global_{}_{} ".format(gws[0], gws[1]), ctypes.c_ulong()),
                                2,
                                Array(body=[Constant(x) for x in gws])))
                        gws_arrays[gws] = SymbolRef("global_{}_{} ".format(gws[0], gws[1]))
                    else:
                        control.append(
                            ArrayDef(SymbolRef("global_%d " % gws, ctypes.c_ulong()), 1, Array(body=[Constant(gws)])))
                        gws_arrays[gws] = SymbolRef("global_%d" % gws)

                if lws not in lws_arrays:
                    if(isinstance(lws, tuple)):
                        control.append(
                            ArrayDef(
                                SymbolRef("local_{}_{} ".format(lws[0], lws[1]), ctypes.c_ulong()),
                                2,
                                Array(body=[Constant(x) for x in lws])))
                        lws_arrays[lws] = SymbolRef("local_{}_{} ".format(lws[0], lws[1]))
                    else:
                        control.append(
                            ArrayDef(
                                SymbolRef("local_%s " % lws, ctypes.c_ulong()),
                                1,
                                Array(body=[Constant(lws)])
                            )
                        )
                        lws_arrays[lws] = SymbolRef("local_%s" % lws)

                # clSetKernelArg
                for arg_num, arg in enumerate(kernel_func.params):
                    set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                           [SymbolRef(kernel_func.name.name),
                                            Constant(arg_num),
                                            Constant(ctypes.sizeof(cl.cl_mem)),
                                            Ref(SymbolRef(arg.name))])
                    control.append(BitOrAssign(error_code, set_arg))

                ocl_dims = 1 if isinstance(gws, int) else len(gws)
                # clEnqueueNDRangeKernel
                enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
                                   SymbolRef("queue"), SymbolRef(kernel_func.name), Constant(ocl_dims), NULL(),
                                   gws_arrays[gws], lws_arrays[lws], Constant(0), NULL(), NULL()
                               ])
                enqueue_call = BitOrAssign(error_code, enqueue_call)
                if self.settings.enqueue_iterations > 1:
                    enqueue_call = For(
                            init=Assign(SymbolRef("kernel_pass"), Constant(0)),
                            test=Lt(SymbolRef("kernel_pass"), Constant(self.settings.enqueue_iterations)),
                            incr=PostInc(SymbolRef("kernel_pass")),
                            body=[enqueue_call]
                        )
                control.append(enqueue_call)
                control.append(StringTemplate("""clFinish(queue);"""))

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
            # print(c_file)
            # for f in ocl_files:
            #     print("{}".format(f.codegen()))
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

    def _post_process(self, original, compiled, index_name, **kwargs):
        return self.LazySpecializedKernel(
            py_ast=compiled,
            original=original,
            names=find_names(original),
            index_name=index_name,
            target_names=[stencil.primary_mesh for stencil in original.body if hasattr(stencil, "primary_mesh")],
            _hash=hash(original),
            context=self.context,
            device=self.device,
            settings=self.settings
        )
