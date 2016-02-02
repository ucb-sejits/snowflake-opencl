from __future__ import print_function
import unittest
import numpy as np
import pycl as cl

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
        buffer_in = np.random.random((buffer_size, ))
        for i, x in enumerate(buffer_in):
            # for j, y in enumerate(x):
                buffer_in[i] = i
        print("buffer_in  is {}".format(buffer_in))
        buffer_out = np.zeros_like(buffer_in)

        device = cl.clGetDeviceIDs()[0]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, buffer_in)
        out_buf = NDBuffer(queue, buffer_out)

        sobel_x_component = StencilComponent('arr', SparseWeightArray({(1, ): 1.0}))
        sobel_total = Stencil(
            sobel_x_component,
            'out',
            ((1, -1, 1), ))

        compiler = OpenCLCompiler(ctx)
        sobel_ocl = compiler.compile(sobel_total)
        sobel_ocl(out_buf, in_buf)

        buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
        print("buffer out {}".format(buffer_out))

        for x in range(1, buffer_size-1):
            self.assertEqual(buffer_out[x], buffer_in[x + 1])

        out_evt.wait()
        print("done")

    def test_sobel(self):
        l = np.random.random((1024, 1024))
        lena_out = np.zeros_like(l)

        device = cl.clGetDeviceIDs()[1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, l)
        out_buf = NDBuffer(queue, lena_out)

        sobel_x_component = StencilComponent('arr', WeightArray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        sobel_y_component = StencilComponent('arr', WeightArray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
        sobel_total = Stencil(
            sobel_x_component * sobel_x_component + sobel_y_component * sobel_y_component,
            'out',
            ((1, -1, 1), (1, -1, 1)))

        compiler = OpenCLCompiler(ctx)
        sobel_ocl = compiler.compile(sobel_total)
        sobel_ocl(out_buf, in_buf)

        lena_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, lena_out)
        print(lena_out)
        out_evt.wait()
        print("done")

    def test_3d_jacobi(self):
        buffer_in = np.random.random((128, 128, 128))
        # for j, x in enumerate(buffer_in):
        #     for i, y in enumerate(x):
        #         for k, _ in enumerate(y):
        #             buffer_in[i, j, k] = i * j * k

        buffer_out = np.zeros_like(buffer_in)

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, buffer_in)
        out_buf = NDBuffer(queue, buffer_out)

        sc = StencilComponent(
            'buffer',
            WeightArray([
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 1, 0],
                    [1, 6, 1],
                    [0, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ])
        )

        jacobi_stencil = Stencil(sc, 'out', ((1, -1, 1),) * 3, primary_mesh='out')

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
