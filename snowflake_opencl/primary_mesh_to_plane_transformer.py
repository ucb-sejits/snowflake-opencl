import ast
import re

from ctree.c.nodes import Constant, SymbolRef, FunctionCall, BinaryOp, ArrayRef, Op

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


# noinspection PyPep8Naming
class PrimaryMeshToPlaneTransformer(ast.NodeTransformer):
    def __init__(self, mesh_name, plane_size):
        self.mesh_name = mesh_name
        self.plane_size = plane_size
        self.debug_plane_transformer = False
        super(PrimaryMeshToPlaneTransformer, self).__init__()

    # def visit(self, node):
    #     print("override visit node: {}".format(node.__class__))
    #     return super(PrimaryMeshToPlaneTransformer, self).visit(node)

    def visit_BinaryOp(self, node):
        if isinstance(node.op, Op.ArrayRef):
            if self.debug_plane_transformer:
                print("in binary op: op is {}".format(node.op.__class__))
            if isinstance(node.left, SymbolRef):
                if node.left.name is self.mesh_name:
                    if self.debug_plane_transformer:
                        print("found mesh name {}, right is {}".format(self.mesh_name, type(node.right)))
                    if isinstance(node.right, FunctionCall) and isinstance(node.right.func, SymbolRef):
                        if self.debug_plane_transformer:
                            print("Function call is {}".format(node.right.func))
                        m = re.match('encode(\d+)_(\d+)_(\d+)', node.right.func.name)
                        if m:
                            if self.debug_plane_transformer:
                                print("got encode {} {} {}".format(m.group(1), m.group(2), m.group(3)))
                            func = node.right
                            if isinstance(func.args[0], BinaryOp) and isinstance(func.args[0].op, Op.Add):
                                add = func.args[0]
                                if isinstance(add.right, Constant):
                                    if self.debug_plane_transformer:
                                        print("index_0 offset is {}".format(add.right.value))

                                    new_node = ArrayRef(
                                        SymbolRef("plane_{}".format(1+add.right.value)),
                                        FunctionCall(
                                            func=SymbolRef("encode{}_{}".format(
                                                self.plane_size[0], self.plane_size[1])),
                                            args=["local_{}".format(x) for x in func.args[1:]]
                                        )
                                    )
                                    return new_node

        return BinaryOp(
            left=self.visit(node.left),
            op=node.op,
            right=self.visit(node.right)
        )
