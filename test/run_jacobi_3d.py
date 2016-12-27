from __future__ import print_function
import numpy as np
import pycl as cl

from snowflake_opencl.pencil_compiler import PencilCompiler
from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil

from snowflake_opencl.compiler import NDBuffer, OpenCLCompiler

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


if __name__ == '__main__':
    size = 10
    import logging
    logging.basicConfig(level=20)

    np.random.seed(0)

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

    compiler = PencilCompiler(ctx, device)
    jacobi_operator = compiler.compile(jacobi_stencil)
    jacobi_operator(out_buf, in_buf)

    buffer_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, buffer_out)
    out_evt.wait()

    print("Input " + "=" * 80)
    print_mesh(buffer_in)
    print("Output" + "=" * 80)
    print_mesh(buffer_out)

    # print("in buf\n" + buffer_in.ary[0,0,0:10])
    # print("out buf\n" + buffer_out[1][1][0:10])
    print("done")
