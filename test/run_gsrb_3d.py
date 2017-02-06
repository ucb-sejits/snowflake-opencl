from __future__ import print_function

import copy
import sys

import argparse
import random

import math
import numpy as np
import pycl as cl
import time

# from snowflake_opencl.pencil_compiler import PencilCompiler
from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil, RectangularDomain, StencilGroup

from snowflake_opencl.compiler import OpenCLCompiler
from snowflake_opencl.nd_buffer import NDBuffer
from snowflake_opencl.settings import Settings

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


if __name__ == '__main__':
    sys.setrecursionlimit(1500)

    parser = argparse.ArgumentParser()
    Settings.add_settings_parsers(parser)
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
        pencil_kernel_size_threshold=args.pencil_kernel_size_threshold,
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

    initial_buffer = np.random.random((size, size, size)).astype(use_type)

    # initial_buffer = np.ones((size, size, size)).astype(use_type)
    # for j, x in enumerate(initial_buffer):
    #     for i, y in enumerate(x):
    #         for k, _ in enumerate(y):
    #             initial_buffer[i, j, k] = 1.0  # float(i * j * k)

    queue = cl.clCreateCommandQueue(ctx)

    buffer_2 = initial_buffer.copy()
    in_buf_2 = NDBuffer(queue, buffer_2)

    sc = StencilComponent(
        'mesh',
        WeightArray(weight_array)
    )

    # red_iteration_space
    #        / A /   / A /   /
    #       /   / B /   / B /
    #      / A /   / A /   /
    #     /   / B /   / B /
    #
    #        /   / C /   / C /
    #       / D /   / D /   /
    #      /   / C /   / C /
    #     / D /   / D /   /
    #
    red_iteration_space_a = RectangularDomain(((1, -2, 2), (1, -2, 2), (1, -2, 2)))
    red_iteration_space_b = RectangularDomain(((2, -1, 2), (2, -1, 2), (1, -2, 2)))
    red_iteration_space_c = RectangularDomain(((2, -1, 2), (1, -2, 2), (2, -1, 2)))
    red_iteration_space_d = RectangularDomain(((1, -2, 2), (2, -1, 2), (2, -1, 2)))

    red_iteration_space = red_iteration_space_a + red_iteration_space_b +\
        red_iteration_space_c + red_iteration_space_d
    red_stencil = Stencil(sc, 'mesh', red_iteration_space, primary_mesh='mesh')

    # black_iteration_space
    #        /   / E /   / E /
    #       / F /   / F /   /
    #      /   / E /   / E /
    #     / F /   / F /   /
    #
    #        / G /   / G /   /
    #       /   / H /   / H /
    #      / G /   / G /   /
    #     /   / H /   / H /
    #
    black_iteration_space_e = RectangularDomain(((1, -2, 2), (1, -2, 2), (2, -1, 2)))
    black_iteration_space_f = RectangularDomain(((2, -1, 2), (2, -1, 2), (2, -1, 2)))
    black_iteration_space_g = RectangularDomain(((2, -1, 2), (1, -2, 2), (1, -2, 2)))
    black_iteration_space_h = RectangularDomain(((1, -2, 2), (2, -1, 2), (1, -2, 2)))

    black_iteration_space = black_iteration_space_e + black_iteration_space_f +\
        black_iteration_space_g + black_iteration_space_h
    black_stencil = Stencil(sc, 'mesh', black_iteration_space, primary_mesh='mesh')

    gsrb_stencil = StencilGroup([red_stencil, black_stencil])

    if run_no_pencil:
        buffer_1 = initial_buffer.copy()
        in_buf_1 = NDBuffer(queue, buffer_1)

        no_pencil_settings = copy.copy(settings)
        no_pencil_settings.label = "no-pencil"
        no_pencil_settings.pencil_kernel_size_threshold = sys.maxint
        compiler = OpenCLCompiler(ctx, device, no_pencil_settings)

        jacobi_operator = compiler.compile(gsrb_stencil)

        start_time = time.time()

        out_evt = None
        for _ in range(iterations):
            jacobi_operator(in_buf_1)
            buffer_1, out_evt = cl.buffer_to_ndarray(queue, in_buf_1.buffer, buffer_1)

        out_evt.wait()

        end_time = time.time()

        if show_mesh:
            print("Input " + "=" * 80)
            print_mesh(initial_buffer)
            print("Output" + "=" * 80)
            print_mesh(buffer_1)

        # import subprocess
        # subprocess.call(["ctree", "-cc"])

    pencil_compiler = OpenCLCompiler(ctx, device, settings)
    jacobi_operator_pencil = pencil_compiler.compile(gsrb_stencil)

    start_time_pencil = time.time()

    out_evt = None
    for _ in range(iterations):
        jacobi_operator_pencil(in_buf_1)
        buffer_2, out_evt = cl.buffer_to_ndarray(queue, in_buf_2.buffer, buffer_2)

    out_evt.wait()

    end_time_pencil = time.time()

    if show_mesh:
        print("Input " + "=" * 80)
        print_mesh(initial_buffer)
        print("Output" + "=" * 80)
        print_mesh(buffer_2)

    if run_no_pencil:
        if test_method == "numpy":
            np.testing.assert_array_almost_equal(buffer_1, buffer_2, decimal=4)
        elif test_method == "python":
            differences = 0
            values_compared = 0
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        values_compared += 1
                        if buffer_1.ary[x, y, z] - buffer_2.ary[x, y, z] > 0.0001:
                            differences += 1
                            computed = buffer_1[x+1, y, z] + buffer_1[x-1, y, z] + \
                                buffer_1[x, y+1, z] + buffer_1[x, y-1, z] + \
                                buffer_1[x, y, z+1] + buffer_1[x, y, z-1] + \
                                (-6.0 * buffer_1[x, y, z])

                            print("computed_value {} {:10.4f} regular {:10.4f} pencil {:10.4f} delta {:10.4f}".format(
                                (x, y, z), computed, buffer_1.ary[x, y, z], buffer_2.ary[x, y, z],
                                buffer_1.ary[x, y, z] - buffer_2.ary[x, y, z]
                            )
                            )

                            little_mesh = buffer_1[x-1:x+2, y-1:y+2, z-1:z+2]
                            print_mesh(little_mesh)
            print("Total differences: {} out of {}".format(differences, values_compared))

    # the following times include compilation and are confusing
    # print("compiler        done in {:10.5f} seconds".format((end_time - start_time) / iterations))
    # print("pencil_compiler done in {:10.5f} seconds".format((end_time_pencil - start_time_pencil) / iterations))
