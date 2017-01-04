import ast
import re

from ctree.c.nodes import Constant, SymbolRef, FunctionCall, BinaryOp, Op

__author__ = 'Chick Markley, Seunghwan Choi, Dorthy Luu'


# noinspection PyPep8Naming
class PrimaryMeshToRegisterTransformer(ast.NodeTransformer):
    def __init__(self, mesh_name, plane_size, settings):
        self.mesh_name = mesh_name
        self.plane_size = plane_size
        self.debug_plane_transformer = False
        self.optimize_plane_offsets = settings.use_plane_offsets
        self.stage = 0
        self.neighbor_id = 0
        self.plane_source_id = 0
        self.plane_offsets = []
        super(PrimaryMeshToRegisterTransformer, self).__init__()

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
                                if len(func.args) == 3:
                                    add_0 = func.args[0]
                                    add_1 = func.args[1]
                                    add_2 = func.args[2]
                                    if isinstance(add_0, BinaryOp) and isinstance(add_0.op, Op.Add) and \
                                            isinstance(add_1, BinaryOp) and isinstance(add_1.op, Op.Add) and \
                                            isinstance(add_2, BinaryOp) and isinstance(add_2.op, Op.Add):
                                        if isinstance(add_0.right, Constant) and \
                                                isinstance(add_1.right, Constant) and \
                                                isinstance(add_2.right, Constant):
                                            const_0 = add_0.right.value
                                            const_1 = add_1.right.value
                                            const_2 = add_2.right.value

                                            if const_1 == 0 and const_2 == 0:
                                                if const_0 == -1:
                                                    return SymbolRef("register_0")
                                                elif const_0 == 0:
                                                    return SymbolRef("register_1")
                                                elif const_0 == 1:
                                                    return SymbolRef("register_2")

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
        return new_node_1
