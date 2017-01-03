from __future__ import print_function

import argparse
import numpy as np
import pycl as cl
import time

import sys

from snowflake_opencl.pencil_compiler import PencilCompiler
from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil

from snowflake_opencl.compiler import OpenCLCompiler
from snowflake_opencl.nd_buffer import NDBuffer
from snowflake_opencl.settings import Settings

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, help="mesh edge size")
    parser.add_argument("-t", "--test-method", type=str)
    parser.add_argument("-i", "--iterations", type=int)
    parser.add_argument("-lm", "--use-local-mem", action="store_true")
    parser.add_argument("-po", "--use-plane-offsets", action="store_true")
    args = parser.parse_args()

    size = args.size
    settings = Settings(args.use_local_mem, args.use_plane_offsets)
    test_method = args.test_method
    iterations = args.iterations
    import logging
    # logging.basicConfig(level=20)

    np.random.seed(0)

    buffer_in = np.random.random((size, size, size)).astype(np.float32)
    # buffer_in = np.ones((size, size, size)).astype(np.float32)
    # for j, x in enumerate(buffer_in):
    #     for i, y in enumerate(x):
    #         for k, _ in enumerate(y):
    #             buffer_in[i, j, k] = 1.0  # float(i * j * k)

    buffer_out = np.zeros_like(buffer_in)
    buffer_out_pencil = np.zeros_like(buffer_in)

    device = cl.clGetDeviceIDs()[-1]
    ctx = cl.clCreateContext(devices=[device])

    queue = cl.clCreateCommandQueue(ctx)

    in_buf = NDBuffer(queue, buffer_in)
    out_buf = NDBuffer(queue, buffer_out)
    out_buf_pencil = NDBuffer(queue, buffer_out_pencil)

    sc = StencilComponent(
        'mesh',
        WeightArray([
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [1, -6, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ])
    )

    jacobi_stencil = Stencil(sc, 'out', ((1, size-1, 1),) * 3, primary_mesh='out')

    compiler = OpenCLCompiler(ctx)

    jacobi_operator = compiler.compile(jacobi_stencil)

    start_time = time.time()

    for _ in range(iterations):
        jacobi_operator(out_buf, in_buf)
        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)

    out_evt.wait()

    end_time = time.time()

    print("Input " + "=" * 80)
    print_mesh(buffer_in)
    print("Output" + "=" * 80)
    print_mesh(buffer_out)

    pencil_compiler = PencilCompiler(ctx, device, settings)
    jacobi_operator_pencil = pencil_compiler.compile(jacobi_stencil)

    start_time_pencil = time.time()

    for _ in range(iterations):
        jacobi_operator_pencil(out_buf_pencil, in_buf)
        buffer_out_pencil, out_evt = cl.buffer_to_ndarray(queue, out_buf_pencil.buffer, buffer_out_pencil)

    out_evt.wait()

    end_time_pencil = time.time()

    # print("Input " + "=" * 80)
    # print_mesh(buffer_in)
    # print("Output" + "=" * 80)
    # print_mesh(buffer_out_pencil)

    if test_method == "numpy":
        np.testing.assert_array_almost_equal(buffer_out, buffer_out_pencil, decimal=4)
    elif test_method == "python":
        differences = 0
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if out_buf.ary[x, y, z] - out_buf_pencil.ary[x, y, z] > 0.0001:
                        differences += 1
                        computed = buffer_in[x+1, y, z] + buffer_in[x-1, y, z] + \
                            buffer_in[x, y+1, z] + buffer_in[x, y-1, z] + \
                            buffer_in[x, y, z+1] + buffer_in[x, y, z-1] + \
                            (-6.0 * buffer_in[x, y, z])

                        print("computed_value {} {:10.4f} regular {:10.4f} pencil {:10.4f} delta {:10.4f}".format(
                            (x, y, z), computed, out_buf.ary[x, y, z], out_buf_pencil.ary[x, y, z],
                            out_buf.ary[x, y, z] - out_buf_pencil.ary[x, y, z]
                        )
                        )

                        little_mesh = buffer_in[x-1:x+2, y-1:y+2, z-1:z+2]
                        print_mesh(little_mesh)
        print("Total differences: {}".format(differences))

    print("compiler        done in {:10.5f} seconds".format((end_time - start_time) / iterations))
    print("pencil_compiler done in {:10.5f} seconds".format((end_time_pencil - start_time_pencil) / iterations))
