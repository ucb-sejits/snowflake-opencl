from __future__ import print_function
import unittest
import numpy as np
import pycl as cl

from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil, SparseWeightArray

from snowflake_opencl.compiler import NDBuffer, OpenCLCompiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestSimpleStencils(unittest.TestCase):
    def test_1d_translate(self):
        """
        slides over every element in a 1-d array 1 element to the left
        :return:
        """
        buffer_size = 16
        buffer_in = np.random.random((buffer_size, )).astype(np.float32)
        for i, x in enumerate(buffer_in):
            # for j, y in enumerate(x):
                buffer_in[i] = i
        print("buffer_in  is {}".format(buffer_in))
        buffer_out = np.zeros_like(buffer_in)

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, buffer_in)
        out_buf = NDBuffer(queue, buffer_out)

        sobel_x_component = StencilComponent('mesh', SparseWeightArray({(1, ): 1.0}))
        sobel_total = Stencil(
            sobel_x_component,
            'out',
            ((1, -1, 1), ))

        compiler = OpenCLCompiler(ctx, device)
        sobel_ocl = compiler.compile(sobel_total)
        sobel_ocl(out_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
        print("buffer out {}".format(buffer_out))

        for x in range(1, buffer_size-1):
            self.assertEqual(buffer_out[x], buffer_in[x + 1])

        out_evt.wait()
        print("done")

    def test_sobel(self):
        size = 16
        lena_in = np.random.random((size, size)).astype(np.float32)
        # counter = 1
        # for j, x in enumerate(lena_in):
        #     for i, y in enumerate(x):
        #         lena_in[i, j] = float(counter)
        #     counter += 1

        lena_out = np.zeros_like(lena_in)

        import logging
        logging.basicConfig(level=20)

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, lena_in)
        out_buf = NDBuffer(queue, lena_out)

        sobel_x_component = StencilComponent('mesh', WeightArray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        sobel_y_component = StencilComponent('mesh', WeightArray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        sobel_total = Stencil(
            (sobel_x_component * sobel_x_component) + (sobel_y_component * sobel_y_component),
            'out',
            ((1, -1, 1), (1, -1, 1)))

        compiler = OpenCLCompiler(ctx, device)
        sobel_ocl = compiler.compile(sobel_total)
        sobel_ocl(out_buf, in_buf)

        lena_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, lena_out)
        out_evt.wait()

        print_mesh(lena_in, "lena_in")
        print_mesh(lena_out, "lena_out")
        print("done")

    def test_2d_jacobi(self):
        size = 11
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
        out_buf = NDBuffer(queue, buffer_out)

        sc = StencilComponent(
            'mesh',
            WeightArray([
                    [0, 1.0, 0],
                    [10.0, 0, 100.0],
                    [0, 1000.0, 0],
            ])
        )

        jacobi_stencil = Stencil(sc, 'out', ((1, -1, 1), (1, -1, 1)), primary_mesh='out')

        compiler = OpenCLCompiler(ctx, device)
        jacobi_operator = compiler.compile(jacobi_stencil)
        jacobi_operator(out_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
        print("Input " + "=" * 80)
        for i in range(size-1, -1, -1):
            for j in range(size):
                print("{:7.0f}".format(buffer_in[(i, j)]), end="")
            print()
        print("Output" + "=" * 80)
        for i in range(size-1, -1, -1):
            for j in range(size):
                print("{:7.0f}".format(buffer_out[(i, j)]), end="")
            print()
        out_evt.wait()
        print("done")

    def test_3d_jacobi(self):
        size = 34
        import logging
        logging.basicConfig(level=20)

        buffer_in = np.random.random((size, size, size)).astype(np.float32)
        for j, x in enumerate(buffer_in):
            for i, y in enumerate(x):
                for k, _ in enumerate(y):
                    buffer_in[i, j, k] = 1.0  # float(i * j * k)

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

        compiler = OpenCLCompiler(ctx, device)
        jacobi_operator = compiler.compile(jacobi_stencil)
        jacobi_operator(out_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
        out_evt.wait()

        print("Input " + "=" * 80)
        print_mesh(buffer_in)
        print("Output" + "=" * 80)
        print_mesh(buffer_out)
        print("done")
