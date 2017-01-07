from __future__ import print_function

import sys

import argparse
import random

import math
import numpy as np
import pycl as cl
import time

from snowflake_opencl.pencil_compiler import PencilCompiler
from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil

from snowflake_opencl.compiler import OpenCLCompiler
from snowflake_opencl.nd_buffer import NDBuffer
from snowflake_opencl.settings import Settings

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


if __name__ == '__main__':
    sys.setrecursionlimit(1500)

    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, help="mesh edge size")
    parser.add_argument("-t", "--test-method", type=str, default="none")
    parser.add_argument("-rnp", "--run-no-pencil", action="store_true")
    parser.add_argument("-i", "--iterations", type=int, default=1)
    parser.add_argument("-lm", "--use-local-mem", action="store_true")
    parser.add_argument("-lr", "--use-local-register", action="store_true")
    parser.add_argument("-po", "--use-plane-offsets", action="store_true")
    parser.add_argument("-sm", "--show-mesh", action="store_true")
    parser.add_argument("-uk", "--unroll-kernel", action="store_true")
    parser.add_argument("-ff", "--force-float", action="store_true")
    parser.add_argument("-sgc", "--show-generated-code", action="store_true")
    parser.add_argument("-sop", "--set-operator", type=str,
                        help='one of 7pt, 13pt, 3x3x3, 5x5x5')
    parser.add_argument("-flws", "--force-local-work-size", type=str)
    parser.add_argument("-tl", "--timer-label", type=str, default="opencl pencil test")
    parser.add_argument("-ei", "--enqueue-iterations", type=int)
    parser.add_argument("-rfbf", "--remove-for-body-fence", action="store_true")
    args = parser.parse_args()

    force_local_work_size = None
    if args.force_local_work_size:
        force_local_work_size = tuple(int(x) for x in args.force_local_work_size.split(","))

    settings = Settings(
        use_local_mem=args.use_local_mem,
        use_plane_offsets=args.use_plane_offsets,
        enqueue_iterations=args.enqueue_iterations,
        use_local_register=args.use_local_register,
        unroll_kernel=args.unroll_kernel,
        force_local_work_size=force_local_work_size,
        remove_for_body_fence=args.remove_for_body_fence,
        label=args.timer_label,
    )

    test_method = args.test_method
    iterations = args.iterations
    show_mesh = args.show_mesh
    run_no_pencil = args.run_no_pencil
    force_float = args.force_float

    if args.show_generated_code:
        import logging
        logging.basicConfig(level=20)

    np.random.seed(0)

    device = cl.clGetDeviceIDs()[-1]
    ctx = cl.clCreateContext(devices=[device])

    use_type = np.float32
    if "cl_khr_fp64" in device.extensions and not force_float:
        use_type = np.double

    def rr():
        return random.random()

    ghost_size = 1
    weight_array = None
    if not args.set_operator:
        ghost_size = 1
        weight_array = [
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
        ]
    elif args.set_operator == "7pt":
        ghost_size = 1
        weight_array = np.array([
            [
                [0, 0, 0],
                [0, rr(), 0],
                [0, 0, 0],
            ],
            [
                [0, rr(), 0],
                [rr(), rr(), rr()],
                [0, rr(), 0],
            ],
            [
                [0, 0, 0],
                [0, rr(), 0],
                [0, 0, 0],
            ],
        ])
    elif args.set_operator == "13pt":
        ghost_size = 2
        weight_array = [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, rr(), 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, rr(), 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, rr(), 0, 0],
                [0, 0, rr(), 0, 0],
                [rr(), rr(), rr(), rr(), rr()],
                [0, 0, rr(), 0, 0],
                [0, 0, rr(), 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, rr(), 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, rr(), 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    elif args.set_operator == "3x3x3":
        ghost_size = 1
        weight_array = np.random.random((3, 3, 3))
    elif args.set_operator == "5x5x5":
        ghost_size = 2
        weight_array = np.random.random((5, 5, 5))
    else:
        print(parser.usage)
        exit(1)

    power = math.log(args.size, 2)
    if power - int(power) > 0:
        print("mesh size {} must be a power of 2".format(args.size))
        print(parser.usage)
        exit(1)

    size = args.size + (2 * ghost_size)

    buffer_in = np.random.random((size, size, size)).astype(use_type)

    # buffer_in = np.ones((size, size, size)).astype(use_type)
    # for j, x in enumerate(buffer_in):
    #     for i, y in enumerate(x):
    #         for k, _ in enumerate(y):
    #             buffer_in[i, j, k] = 1.0  # float(i * j * k)

    buffer_out = np.zeros_like(buffer_in)
    buffer_out_pencil = np.zeros_like(buffer_in)

    queue = cl.clCreateCommandQueue(ctx)

    in_buf = NDBuffer(queue, buffer_in)
    out_buf = NDBuffer(queue, buffer_out)
    out_buf_pencil = NDBuffer(queue, buffer_out_pencil)

    sc = StencilComponent(
        'mesh',
        WeightArray(weight_array)
    )

    jacobi_stencil = Stencil(sc, 'out', ((ghost_size, size-ghost_size, 1),) * 3, primary_mesh='out')

    if run_no_pencil:
        compiler = OpenCLCompiler(ctx, device, settings)

        jacobi_operator = compiler.compile(jacobi_stencil)

        start_time = time.time()

        out_evt = None
        for _ in range(iterations):
            jacobi_operator(out_buf, in_buf)
            buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)

        out_evt.wait()

        end_time = time.time()

        if show_mesh:
            print("Input " + "=" * 80)
            print_mesh(buffer_in)
            print("Output" + "=" * 80)
            print_mesh(buffer_out)

    pencil_compiler = PencilCompiler(ctx, device, settings)
    jacobi_operator_pencil = pencil_compiler.compile(jacobi_stencil)

    start_time_pencil = time.time()

    out_evt = None
    for _ in range(iterations):
        jacobi_operator_pencil(out_buf_pencil, in_buf)
        buffer_out_pencil, out_evt = cl.buffer_to_ndarray(queue, out_buf_pencil.buffer, buffer_out_pencil)

    out_evt.wait()

    end_time_pencil = time.time()

    if show_mesh:
        print("Input " + "=" * 80)
        print_mesh(buffer_in)
        print("Output" + "=" * 80)
        print_mesh(buffer_out_pencil)

    if run_no_pencil:
        if test_method == "numpy":
            np.testing.assert_array_almost_equal(buffer_out, buffer_out_pencil, decimal=4)
        elif test_method == "python":
            differences = 0
            values_compared = 0
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        values_compared += 1
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
            print("Total differences: {} out of {}".format(differences, values_compared))

    # the following times include compilation and are confusing
    # print("compiler        done in {:10.5f} seconds".format((end_time - start_time) / iterations))
    # print("pencil_compiler done in {:10.5f} seconds".format((end_time_pencil - start_time_pencil) / iterations))
