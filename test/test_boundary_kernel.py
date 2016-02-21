from __future__ import print_function
import unittest
import numpy as np
import pycl as cl

from snowflake.nodes import StencilComponent, Stencil, SparseWeightArray

from snowflake_opencl.compiler import NDBuffer, OpenCLCompiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestBoundaryStencils(unittest.TestCase):
    def test_v2_3d_face_kernel(self):
        size = 4

        mesh = np.zeros([size, size, size], dtype=np.float32)
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                for k in range(1, size - 1):
                    mesh[i, j, k] = 1.0

        print("buffer_in  is {}".format(mesh))

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, mesh)

        boundary_component = StencilComponent(
            'mesh',
            SparseWeightArray({
                (1, 0, 0): 100.0,
                (2, 0, 0): 1.0,
            })
        )
        boundary_stencil = Stencil(
            boundary_component,
            'mesh',
            ((size-1, size, 1), (0, size, 1), (0, size, 1)))

        compiler = OpenCLCompiler(ctx)
        sobel_ocl = compiler.compile(boundary_stencil)
        sobel_ocl(in_buf)
        print(sobel_ocl.arg_spec)

        mesh, out_evt = cl.buffer_to_ndarray(queue, in_buf.buffer, mesh)

        out_evt.wait()

        print("buffer out {}".format(mesh))
        print("linear {}".format(mesh.reshape((size**3,))))

        print("done")

    def test_v2_2d_face_kernel(self):
        size = 6

        import logging
        logging.basicConfig(level=20)

        mesh = np.zeros([size, size], dtype=np.float32)
        count = 0
        for i in range(size):
            for j in range(size):
                mesh[i, j] = count
                count += 1

        for i in range(size-1, -1, -1):
            for j in range(size):
                print(" {:5d}".format(int(mesh[i, j])), end="")
            print()

        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])
        queue = cl.clCreateCommandQueue(ctx)

        in_buf = NDBuffer(queue, mesh)

        boundary_component = StencilComponent(
            'mesh',
            SparseWeightArray({
                (-1, 0): 100.0,
                (-2, 0): 1.0,
            })
        )
        boundary_stencil = Stencil(
            boundary_component,
            'mesh',
            ((size-1, size, 1), (1, size-1, 1)))

        compiler = OpenCLCompiler(ctx)
        sobel_ocl = compiler.compile(boundary_stencil)
        sobel_ocl(in_buf)
        print(sobel_ocl.arg_spec)

        mesh, out_evt = cl.buffer_to_ndarray(queue, in_buf.buffer, mesh)

        out_evt.wait()

        for i in range(size-1, -1, -1):
            for j in range(size):
                print(" {:5d}".format(int(mesh[i, j])), end="")
            print()
        print("\n\n")
        buf2 = mesh.reshape((size**2,))
        for i in range(len(buf2)):
            print(" {:4d}".format(i), end="")
        print()
        for i in range(len(buf2)):
            print(" {:4d}".format(int(buf2[i])), end="")
        print()
        print("done")
