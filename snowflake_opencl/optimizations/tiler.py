from __future__ import print_function

import ast
import ctree.c.nodes as ctree_nodes
from ctree.c.nodes import SymbolRef, Op

__author__ = 'Chick Markley chick@berkeley.edu U.C. Berkeley'


class TilingOptimizer(ast.NodeTransformer):
    """
    Convert the default linear 1d opencl tiles (local_work_sizes) int
    n-d tiles that are optimally sized.
    Try to make sides integer multiples of iterations space
    Try to maximize volume where it makes sense
    Add conditionals to stencil computations in those cases where
    they may run outside of the iterations space (i.e. cases where
    an integer multiple of a dimension cannot be chosen
    """
    def __init__(self, reference_shape):
        self.reference_shape = reference_shape
        print("In TilingOptimizer Init, reference shape is {}".format(self.reference_shape))

    def visit(self, node):
        return super(TilingOptimizer, self).visit(node)

    def visit_BinaryOp(self, node):
        print("Assign node found {} op {} assign {}".format(node, node.op, node.op._c_str == "="))
        if isinstance(node.op, Op.Assign) and isinstance(node.left, SymbolRef):
            print("Got us a symbol ref {}".format(node.left.name))

        return node

