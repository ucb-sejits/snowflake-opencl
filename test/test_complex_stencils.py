from __future__ import print_function
import unittest
import numpy as np
import pycl as cl

from snowflake.nodes import StencilComponent, WeightArray, Stencil, SparseWeightArray, StencilGroup, RectangularDomain

from snowflake_opencl.compiler import NDBuffer, OpenCLCompiler
from snowflake_opencl.util import print_mesh

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestComplexStencils(unittest.TestCase):
    def test_2d_gsrb(self):
        size = 10
        import logging
        logging.basicConfig(level=20)

        buffer_in = np.random.random((size, size)).astype(np.float32)
        counter = 1
        for j, x in enumerate(buffer_in):
            for i, y in enumerate(x):
                buffer_in[i, j] = float(counter)
            counter += 1

        buffer_out = np.zeros_like(buffer_in)

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, buffer_in)

        sc = StencilComponent(
            'mesh',
            WeightArray([[10.0]])
            # WeightArray([
            #         [0, 1.0, 0],
            #         [10.0, 0, 100.0],
            #         [0, 1000.0, 0],
            # ])
        )

        # red_iteration_space1 = RectangularDomain(((1, 1), (-2, -2), (2, 2)))
        red_iteration_space1 = RectangularDomain(((1, -2, 2), (1, -2, 2)))
        red_iteration_space2 = RectangularDomain(((2, -1, 2), (2, -1, 2)))
        red_iteration_space = red_iteration_space1 + red_iteration_space2
        # red_iteration_space = red_iteration_space1
        red_stencil = Stencil(sc, 'mesh', red_iteration_space, primary_mesh='out')

        black_iteration_space1 = RectangularDomain(((1, -2, 2), (2, -1, 2)))
        black_iteration_space2 = RectangularDomain(((2, -1, 2), (1, -2, 2)))
        black_iteration_space = black_iteration_space1 + black_iteration_space2
        black_stencil = Stencil(sc, 'mesh', black_iteration_space, primary_mesh='out')

        stencil_group = StencilGroup([red_stencil, black_stencil])
        # stencil_group = StencilGroup([red_stencil])

        compiler = OpenCLCompiler(ctx)
        jacobi_operator = compiler.compile(stencil_group)
        jacobi_operator(in_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, in_buf.buffer, buffer_out)
        out_evt.wait()

        print_mesh(buffer_in, "Input Buffer "+"="*60)
        print_mesh(buffer_out, "Ouput Buffer "+"="*60)
        print("done")

    def test_3d_gsrb(self):
        size = 4
        import logging
        logging.basicConfig(level=20)

        buffer_in = np.random.random((size, size, size)).astype(np.float32)
        # for j, x in enumerate(buffer_in):
        #     for i, y in enumerate(x):
        #         for k, _ in enumerate(y):
        #             buffer_in[i, j, k] = 1.0  # float(i * j * k)

        buffer_out = np.zeros_like(buffer_in)

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, buffer_in)
        out_buf = NDBuffer(queue, buffer_out)

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
        jacobi_operator(out_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
        print("Input " + "=" * 80)
        print(buffer_in)
        print("Output" + "=" * 80)
        print(buffer_out)
        out_evt.wait()
        print("done")
