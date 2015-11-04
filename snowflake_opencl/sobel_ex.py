from snowflake.nodes import *
import numpy as np
from compiler import OpenCLCompiler, NDBuffer
import pycl as cl
# from scipy import misc

# l = np.tile(misc.lena(), (2, 2))

l = np.ndarray((1024, 1024))
lena_out = np.zeros_like(l)

device = cl.clGetDeviceIDs()[0]
ctx = cl.clCreateContext(devices=[device])
queue = cl.clCreateCommandQueue(ctx)

# in_buf, in_evt = cl.buffer_from_ndarray(queue, l)
in_buf = NDBuffer(queue, l)
# out_buf, out_evt = cl.buffer_from_ndarray(queue, lena_out)
out_buf = NDBuffer(queue, lena_out)
# cl.clWaitForEvents(in_evt, out_evt)

sobel_x_component = StencilComponent('arr', WeightArray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
sobel_y_component = StencilComponent('arr', WeightArray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
sobel_total = Stencil(sobel_x_component * sobel_x_component + sobel_y_component * sobel_y_component, 'out', ((1, -1, 1), (1, -1, 1)))

compiler = OpenCLCompiler(ctx)
sobel_ocl = compiler.compile(sobel_total)
sobel_ocl(out_buf, in_buf)

lena_out, out_evt = cl.buffer_to_ndarray(queue, out_buf.buffer, lena_out)
out_evt.wait()
print("done")