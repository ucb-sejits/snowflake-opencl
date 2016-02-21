from ctree.c.macros import NULL
from ctree.c.nodes import Mod, Div, Constant, Add, Mul, SymbolRef, ArrayDef, FunctionDecl, \
    Assign, Array, FunctionCall, Ref, Return, CFile
from ctree.templates.nodes import StringTemplate
import operator
import ctypes
import pycl as cl

__author__ = 'dorthy luu'


def flattened_to_multi_index(flattened_id_symbol, shape, multipliers=None, offsets=None):
    # flattened_id should be a node
    # offsets applied after multipliers

    body = []
    ndim = len(shape)
    full_size = reduce(operator.mul, shape, 1)
    for i in reversed(range(ndim)):
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


def local_work_size(gws):
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
