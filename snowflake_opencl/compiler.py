import operator
import ctypes
import pycl as cl
from ctree.c.nodes import Constant, Assign, SymbolRef, FunctionCall, Div, Add, Mod, Mul, FunctionDecl, MultiNode
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
from ctree.types import get_ctype
from ctree.transformations import PyBasicConversions
from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.nodes import Project
from snowflake._compiler import find_names
from snowflake.compiler_utils import generate_encode_macro
from snowflake.stencil_compiler import Compiler, CCompiler

__author__ = 'dorthyluu'


class NDBuffer(object):
    def __init__(self, queue, ary, blocking=True):
        self.ary = ary
        self.shape = ary.shape
        self.dtype = ary.dtype
        self.buffer, evt = cl.buffer_from_ndarray(queue, ary)
        if blocking:
            evt.wait()


class OpenCLCompiler(Compiler):

    def __init__(self, context):
        super(OpenCLCompiler, self).__init__()
        self.context = context

    BlockConverter = CCompiler.BlockConverter
    IndexOpToEncode = CCompiler.IndexOpToEncode

    class IterationSpaceExpander(CCompiler.IterationSpaceExpander):
        def visit_IterationSpace(self, node):
            node = self.generic_visit(node)

            make_low = lambda low, dim: low if low >= 0 else self.reference_array_shape[dim] + low
            make_high = lambda high, dim: high if high > 0 else self.reference_array_shape[dim] + high

            total_work_dims, total_strides, total_lows = [], [], []

            for space in node.space.spaces:
                lows = tuple(make_low(low, dim) for dim, low in enumerate(space[0]))
                highs = tuple(make_high(high, dim) for dim, high in enumerate(space[1]))
                strides = space[2]
                work_dims = []

                for dim, (high, low, stride) in reversed(list(enumerate(zip(lows, highs, strides)))):
                    work_dims.append((high - low + stride - 1) / stride)

                total_work_dims.append(tuple(work_dims))
                total_strides.append(strides)
                total_lows.append(lows)

            parts = []
            # get_global_id(0)
            parts.append(Assign(SymbolRef("global_id", ctypes.c_ulong()),
                                FunctionCall(SymbolRef("get_global_id"), [Constant(0)])))
            # initialize index variables
            parts.extend(
                SymbolRef("{}_{}".format(self.index_name, dim), ctypes.c_ulong())
                for dim in range(len(self.reference_array_shape)))

            # calculate each index inline
            for space in range(len(node.space.spaces)):
                indices = flattened_to_multi_index(SymbolRef("global_id"),
                                                   total_work_dims[space],
                                                   total_strides[space],
                                                   total_lows[space])
                for dim in range(len(self.reference_array_shape)):
                    parts.append(Assign(SymbolRef("{}_{}".format(self.index_name, dim)), indices[dim]))
                parts.extend(node.body)
            return MultiNode(parts)

    class ConcreteSpecializedKernel(ConcreteSpecializedFunction):
        def __init__(self, context, global_work_size, local_work_size):
            self.context = context
            self.gws = global_work_size
            self.lws = local_work_size

            self.kernels = []
            super(OpenCLCompiler.ConcreteSpecializedKernel, self).__init__()

        def finalize(self, entry_point_name, project_node, entry_point_typesig):
            for file in project_node.files:
                source_code = file.codegen()
                # file._compile(source_code)
                kernel = cl.clCreateProgramWithSource(self.context, source_code).build()["stencil_kernel"]
                self.kernels.append(kernel)
            return self

        def __call__(self, *args, **kwargs):
            queue = cl.clCreateCommandQueue(self.context)
            true_args = [arg.buffer if isinstance(arg, NDBuffer) else arg for arg in args]
            events = []
            for kernel in self.kernels:
                # run_evt = kernel.kernel(*kernel_args).on(self.queue, gsize=kernel.gsize, lsize=kernel.lsize)
                run_evt = kernel(*true_args).on(queue, gsize=self.gws, lsize=self.lws)
                events.append(run_evt)
            cl.clWaitForEvents(*events)

    class LazySpecializedKernel(CCompiler.LazySpecializedKernel):
        def __init__(self, py_ast=None, names=None, target_names=('out',), index_name='index',
                     _hash=None, context=None):

            self.__hash = _hash if _hash is not None else hash(py_ast)
            self.names = names
            self.target_names = target_names
            self.index_name = index_name

            super(OpenCLCompiler.LazySpecializedKernel, self).__init__(
                py_ast, names, target_names, index_name, _hash
            )

            self.parent_cls = OpenCLCompiler
            self.context = context
            self.global_work_size = 0
            self.local_work_size = 0

        def transform(self, tree, program_config):
            subconfig, tuning_config = program_config
            name_shape_map = {name: arg.shape for name, arg in subconfig.items()}
            shapes = set(name_shape_map.values())

            self.parent_cls.IndexOpToEncode(name_shape_map).visit(tree)
            c_tree = PyBasicConversions().visit(tree)

            encode_funcs = []
            for shape in shapes:
                encode_funcs.append(generate_encode_macro('encode'+CCompiler._shape_to_str(shape), shape))

            includes = []
            includes.append(StringTemplate("#pragma OPENCL EXTENSION cl_khr_fp64 : enable"))  # should add this to ctree

            components = []
            # is there a one to one mapping between targets and IterationSpace nodes?
            for target, ispace in zip(self.target_names, c_tree.body):
                shape = subconfig[target].shape
                self.global_work_size = global_work_size(shape, ispace)  # should only do once
                sub = self.parent_cls.IterationSpaceExpander(self.index_name, shape).visit(ispace)
                sub = self.parent_cls.BlockConverter().visit(sub) # changes node to MultiNode
                components.append(sub)

            self.local_work_size = local_work_size(self.global_work_size)

            kernel_func = FunctionDecl(name=SymbolRef("stencil_kernel"),  # 'kernel' is a keyword, embed specific name?
                                       params=[
                                               SymbolRef(name=arg_name, sym_type=get_ctype(
                                                    arg if not isinstance(arg, NDBuffer) else arg.ary.ravel() #hack
                                                ), _global=True) for arg_name, arg in subconfig.items()
                                              ],
                                       defn=components)
            kernel_func.set_kernel()
            ocl_file = OclFile(body=includes + encode_funcs + [kernel_func])
            # print(ocl_file)
            return [ocl_file]


        def finalize(self, transform_result, program_config):
            proj = Project(files=transform_result)
            fn = OpenCLCompiler.ConcreteSpecializedKernel(self.context, self.global_work_size, self.local_work_size)
            return fn.finalize(
                entry_point_name='stencil_kernel',  # not used
                project_node=proj,
                entry_point_typesig=None,
                )

    def _post_process(self, original, compiled, index_name, **kwargs):
        return self.LazySpecializedKernel(
            py_ast=compiled,
            names=find_names(original),
            index_name=index_name,
            target_names=[stencil.primary_mesh for stencil in original.body if hasattr(stencil, "primary_mesh")],
            _hash=hash(original),
            context=self.context
        )


def flattened_to_multi_index(flattened_id_symbol, shape, multipliers=None, offsets=None):
    # flattened_id should be a node
    # offsets applied after multipliers

    body = []
    ndim = len(shape)
    full_size = reduce(operator.mul, shape, 1)
    for i in range(ndim):
        stmt = flattened_id_symbol
        mod_size = reduce(operator.mul, shape[i:], 1)
        div_size = reduce(operator.mul, shape[(i + 1):], 1)
        if mod_size < full_size:
            stmt = Mod(stmt, Constant(mod_size))
        if div_size != 1:
            stmt = Div(stmt, Constant(div_size))
        if multipliers and multipliers[i] != 1:
            stmt = Mul(stmt, Constant(multipliers[i]))
        if offsets and offsets[i] != 0:
            stmt = Add(stmt, Constant(offsets[i]))
        body.append(stmt)
    return body

def global_work_size(array_shape, iteration_space):
    gws = []
    make_low = lambda low, dim: low if low >= 0 else array_shape[dim] + low
    make_high = lambda high, dim: high if high > 0 else array_shape[dim] + high

    for space in iteration_space.space.spaces:
        lows = tuple(make_low(low, dim) for dim, low in enumerate(space[0]))
        highs = tuple(make_high(high, dim) for dim, high in enumerate(space[1]))
        strides = space[2]
        size = 1
        for dim, (high, low, stride) in reversed(list(enumerate(zip(lows, highs, strides)))):
            size *= (high - low + stride - 1) / stride
        gws.append(size)

    if (all(size == gws[0] for size in gws)):
        return gws[0]
    else:
        raise NotImplementedError("Different number of threads per space in IterationSpace not implemented.")

def local_work_size(gws):
    lws = 32
    while gws % lws != 0:
        lws -= 1
    return lws
