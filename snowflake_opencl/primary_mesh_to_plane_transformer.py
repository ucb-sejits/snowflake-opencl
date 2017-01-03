import ast
import ctypes
import re

from ctree.c.nodes import Constant, SymbolRef, FunctionCall, BinaryOp, ArrayRef, Op, Assign

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


# noinspection PyPep8Naming
class PrimaryMeshToPlaneTransformer(ast.NodeTransformer):
    def __init__(self, mesh_name, plane_size):
        self.mesh_name = mesh_name
        self.plane_size = plane_size
        self.debug_plane_transformer = False
        self.stage = 0
        self.neighbor_id = 0
        self.plane_source_id = 0
        self.plane_offsets = []
        super(PrimaryMeshToPlaneTransformer, self).__init__()

    # def visit(self, node):
    #     print("override visit node: {}".format(node.__class__))
    #     return super(PrimaryMeshToPlaneTransformer, self).visit(node)

    def visit_BinaryOp(self, node):
        if self.stage == 0:
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

                                        neighbor_name = "neighbor_{}".format(self.neighbor_id)
                                        self.neighbor_id += 1
                                        neighbor_symbol_init = SymbolRef(neighbor_name, ctypes.c_ulong())
                                        neighbor_symbol = SymbolRef(neighbor_name)
                                        plane_offset = FunctionCall(
                                                func=SymbolRef("encode{}_{}".format(
                                                    self.plane_size[0], self.plane_size[1])),
                                                args=["local_{}".format(x) for x in func.args[1:]]
                                            )
                                        self.plane_offsets.append(Assign(neighbor_symbol_init, plane_offset))
                                        new_node = ArrayRef(
                                            SymbolRef("plane_{}".format(1+add.right.value)),
                                            neighbor_symbol
                                        )
                                        return new_node

        return BinaryOp(
            left=self.visit(node.left),
            op=node.op,
            right=self.visit(node.right)
        )

    # def visit_FunctionCall(self, node):
    #     if self.stage == 1:
    #         m = re.match('encode(\d+)_(\d+)_(\d+)', node.func.name)
    #         if m:
    #             print("got second stage encode {}".format(node.func.name))
    #             plane_source_name = "neighbor_{}".format(self.neighbor_id)
    #             self.plane_source_id += 1
    #             plane_source_symbol_init = SymbolRef(plane_source_name, ctypes.c_ulong())
    #             plane_source_symbol = SymbolRef(plane_source_name)
    #             self.plane_offsets.append(Assign(plane_source_symbol_init, node))
    #             return plane_source_symbol
    #
    #     return node

    def execute(self, node):
        new_node_1 = self.visit(node)
        self.stage = 1
        # new_node_2 = self.visit(new_node_1)
        # return new_node_2
        return new_node_1
