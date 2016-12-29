from __future__ import print_function
from ctree.c.macros import NULL
from ctree.c.nodes import Mod, Div, Constant, Add, Mul, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile
from ctree.templates.nodes import StringTemplate
import operator
import ctypes
import pycl as cl
import sympy

__author__ = 'dorthy luu'


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

def compute_virtual_indexing_parameters(actual_shape, iteration_shape, offset):
    """
    given a mesh of some size, and sub-region of that mesh with a given offset
    and given a 1d position global_index in the global space, compute the coefficients necessary
    to calculate the n-d indices for global_index

    :param actual_shape:
    :param iteration_shape:
    :param offset:
    :return:
    """
    global_1d_size = reduce(operator.mul, actual_shape, 1)
    local_1d_size = reduce(operator.mul, iteration_shape, 1)
    offset_1d_size = reduce(operator.mul, offset, 1)

    tile_number = lambda x: (x - offset_1d_size) / local_1d_size


def get_packed_iterations_shape(array_shape, iteration_space):
    """
    compact the iteration space by fixing highs and lows as necessary and then
    squishing out the strides.

    :param array_shape:
    :param iteration_space:
    :return:
    """
    gws = []

    def make_low(floor, dimension):
        return floor if floor >= 0 else array_shape[dimension] + floor

    def make_high(ceiling, dimension):
        return ceiling if ceiling > 0 else array_shape[dimension] + ceiling

    for space in iteration_space.space.spaces:
        lows = tuple(make_low(low, dim) for dim, low in enumerate(space.low))
        highs = tuple(make_high(high, dim) for dim, high in enumerate(space.high))
        strides = space.stride
        gws.append(
            tuple(
                [(high - low + stride - 1) / stride
                 for (low, high, stride) in reversed(list(zip(lows, highs, strides)))
                ]
            ))

    if all(other_shapes == gws[0] for other_shapes in gws[1:]):
        return gws[0]
    else:
        raise NotImplementedError("Different number of threads per space in IterationSpace not implemented.")


def get_global_work_size(array_shape, iteration_space):
    gws = []

    def make_low(floor, dimension):
        return floor if floor >= 0 else array_shape[dimension] + floor

    def make_high(ceiling, dimension):
        return ceiling if ceiling > 0 else array_shape[dimension] + ceiling

    for space in iteration_space.space.spaces:
        lows = tuple(make_low(low, dim) for dim, low in enumerate(space.low))
        highs = tuple(make_high(high, dim) for dim, high in enumerate(space.high))
        strides = space.stride
        size = 1
        for dim, (low, high, stride) in reversed(list(enumerate(zip(lows, highs, strides)))):
            size *= (high - low + stride - 1) / stride
        gws.append(size)

    if all(size == gws[0] for size in gws):
        return gws[0]
    else:
        raise NotImplementedError("Different number of threads per space in IterationSpace not implemented.")


def get_local_work_size(gws):
    lws = 32
    while gws % lws != 0:
        lws -= 1
    return lws


def generate_control(name, global_size, local_size, kernel_params, kernel, other=None):
    # assumes that all kernels take the same arguments and that they all use the same global and local size!
    defn = [
        ArrayDef(SymbolRef("global", ctypes.c_ulong()), 1, Array(body=[Constant(global_size)])),
        ArrayDef(SymbolRef("local", ctypes.c_ulong()), 1, Array(body=[Constant(local_size)])),
        Assign(SymbolRef("error_code", ctypes.c_int()), Constant(0)),
    ]
    # for kernel in kernels:
    kernel_name = kernel.find(FunctionDecl, kernel=True).name
    for param, num in zip(kernel_params, range(len(kernel_params))):
        if isinstance(param, ctypes.POINTER(ctypes.c_double)):
            set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                   [SymbolRef(kernel_name),
                                    Constant(num),
                                    FunctionCall(SymbolRef("sizeof"), [SymbolRef("cl_mem")]),
                                    Ref(SymbolRef(param.name))])
        else:
            set_arg = FunctionCall(SymbolRef("clSetKernelArg"),
                                   [SymbolRef(kernel_name),
                                    Constant(num),
                                    Constant(ctypes.sizeof(param.type)),
                                    Ref(SymbolRef(param.name))])
        defn.append(set_arg)
    enqueue_call = FunctionCall(SymbolRef("clEnqueueNDRangeKernel"), [
        SymbolRef("queue"), SymbolRef(kernel_name), Constant(1), NULL(),
        SymbolRef("global"), SymbolRef("local"), Constant(0), NULL(), NULL()])
    defn.append(enqueue_call)
    defn.append(StringTemplate("""clFinish(queue);"""))
    defn.append(Return(SymbolRef("error_code")))
    params = [SymbolRef("queue", cl.cl_command_queue()),
              SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel())] + kernel_params
    # params.append(SymbolRef("queue", cl.cl_command_queue()))
    # for kernel in kernels:
    #     params.append(SymbolRef(kernel.find(FunctionDecl, kernel=True).name, cl.cl_kernel()))
    # for param in kernel_params:
    #     if isinstance(param.type, ctypes.POINTER(ctypes.c_double)):
    #         params.append(SymbolRef(param.name, cl.cl_mem()))
    #     else:
    #         params.append(param)
    func = FunctionDecl(ctypes.c_int(), name, params, defn)
    ocl_include = StringTemplate("""
            #include <stdio.h>
            #include <time.h>
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """)
    body = [ocl_include, func]
    if other:
        body.extend(other)
    out_file = CFile(name=name, body=body, config_target='opencl')
    print(out_file)
    return out_file

def print_mesh(mesh, message=None):
    """
    print this mesh, if 3d axes go up the page
    if 2d then standard over and down
    :return:
    """
    if mesh.shape[0] > 18:
        mesh = mesh[-8:, -8:, -8:]

    shape = mesh.shape

    if message:
        print("Mesh print {} shape {}".format(message, shape))

    if len(shape) == 4:
        max_h, max_i, max_j, max_k = shape

        for h in range(max_h):
            print("hyperplane {}".format(h))
            for i in range(max_i-1, -1, -1):
                # print("i  {}".format(i))
                for j in range(max_j-1, -1, -1):
                    print(" "*j*2, end="")
                    for k in range(max_k):
                        print("{:10.6f}".format(mesh[(h, i, j, k)]), end=" ")
                    print()
                print()
            print()
    elif len(shape) == 3:
        max_i, max_j, max_k = shape

        for i in range(max_i-1, -1, -1):
            # print("i  {}".format(i))
            for j in range(max_j-1, -1, -1):
                print(" "*j*2, end="")
                for k in range(max_k):
                    print("{:10.4f}".format(mesh[(i, j, k)]), end=" ")
                print()
            print()
        print()
    elif len(shape) == 2:
        max_i, max_j = shape

        for i in range(max_i):
            # print("i  {}".format(i))
            for j in range(max_j):
                print("{:10.5f}".format(mesh[(i, j)]), end=" ")
            print()
        print()
    else:
        print("I don't know how to print mesh with {} dimensions".format(len(shape)))
