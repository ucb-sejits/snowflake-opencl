from __future__ import print_function
import unittest
import numpy as np
import itertools
from snowflake.nodes import *
from snowflake.vector import Vector
from snowflake.stencil_compiler import *
from snowflake.utils import swap_variables
import pycl as cl

from snowflake_opencl.util import print_mesh

from snowflake.nodes import StencilComponent, WeightArray, Stencil, SparseWeightArray

from snowflake_opencl.compiler import NDBuffer, OpenCLCompiler

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class TestHpgmgApplyOp(unittest.TestCase):
    def test_kernel(self):
        # SIZES = [5, 6, 7, 8]
        SIZES = [4]
        NDIM = [3]
        ITER = 100
        EXPERIMENTS = ["APPLY_OP", "JACOBI"]
        BACKENDS = ["OPENCL", "OPENMP"]
        recordings = {
            i : {j: {} for j in BACKENDS} for i in EXPERIMENTS
        }
        device = cl.clGetDeviceIDs()[-1]
        ctx = cl.clCreateContext(devices=[device])

        def show_recordings(recordings):
            print("Recordings {}".format(recordings))
            reports = recordings.keys()
            for report_name in reports:
                report = recordings[report_name]
                col_names = report.keys()
                first_column_name = col_names[0]
                print("row_names {}".format(row_names))
                row_names = sorted(report[first_column_name].keys())

                print("report {}".format(report_name))
                print('size,{}'.format(",".join(['"{}"'.format(col_name) for col_name in col_names])) )
                for row_name in row_names:
                    print('"2^{}",'.format(row_name[0]), end="")
                    for column in col_names:
                        print("{},".format(report[column][row_name]), end="")
                    print()

        def iterate_component(component):
            body = [component, swap_variables(component, {"mesh":"out", "out":"mesh"})] * (ITER//2)
            return StencilGroup(body)

        def create_apply_op(a, b, h2inv, ndim):
            a_component = a * StencilComponent('mesh',
                                                    SparseWeightArray({Vector.zero_vector(ndim): 1}))
            von_neumann_points = list(Vector.von_neumann_vectors(ndim, radius=1, closed=False))
            weights = {point: 1 for point in von_neumann_points}
            weights[Vector.zero_vector(ndim)] = -len(von_neumann_points)
            b_component = b * h2inv * StencilComponent('mesh', SparseWeightArray(weights))
            return a_component - b_component

        def test_OpenCL(stencil, arrays):
            kern = OpenCLCompiler(ctx).compile(stencil)
            kern(*arrays)
            t = -time.time()
            for i in range(20):
                kern(*arrays)
            t += time.time()
            return t / 20

        def get_stencil(a, b, h2inv, ndim, weight):
            a_x = create_apply_op(a, b, h2inv, ndim)
            b = StencilComponent('rhs_mesh', SparseWeightArray({Vector.zero_vector(ndim): 1}))
            lambda_ref = StencilComponent('lambda_mesh', SparseWeightArray({Vector.zero_vector(ndim): 1}))
            working_ref = StencilComponent('mesh', SparseWeightArray({Vector.zero_vector(ndim): 1}))
            rhs = working_ref + (weight * lambda_ref * (b - a_x))
            return Stencil(rhs, 'out', ((1, -1, 1),) * ndim, primary_mesh='out')

        for input_size, ndim in itertools.product(SIZES, NDIM):
            in_arr = np.random.random((2**input_size + 2,) * ndim)
            rhs_mesh = np.random.random((2**input_size + 2,) * ndim)
            lambda_mesh = np.random.random((2**input_size + 2,) * ndim)
            out = np.empty_like(in_arr)

            queue = cl.clCreateCommandQueue(ctx)
            in_buf = NDBuffer(queue, in_arr)
            out_buf = NDBuffer(queue, out)
            lambda_buf = NDBuffer(queue, lambda_mesh)
            rhs_buf = NDBuffer(queue, rhs_mesh)

            # print(in_arr.shape)

            single_stencil = Stencil(create_apply_op(1, 1, 1, ndim), 'out',  ((1, -1, 1),) * ndim)
            apply_op = iterate_component(single_stencil)
            args = [out_buf, in_buf]
            print("APPLY_OP")
            recordings["APPLY_OP"]["OPENCL"][(input_size, ndim)] = test_OpenCL(apply_op, args)

            smooth_sten = iterate_component(get_stencil(1, 1, 1, ndim, 1))
            # args = [out, lambda_mesh, in_arr, rhs_mesh]
            args = [out_buf, lambda_buf, in_buf, rhs_buf]
            print("Jacobi")
            recordings["JACOBI"]["OPENCL"][(input_size, ndim)] = test_OpenCL(smooth_sten, args)

        print("RECORDINGS")
        show_recordings(recordings)