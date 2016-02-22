from __future__ import print_function
import unittest
import numpy as np
from ctree.c.nodes import Assign, SymbolRef

from snowflake_opencl.util import flattened_to_multi_index

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'


class Test2dFlattening(unittest.TestCase):
    assigns = Assign(
        SymbolRef("index0"),
        flattened_to_multi_index("global_id", shape=(0, 1), multipliers=(1, 1), offsets=(0, 1))
    )
    print("assigns  {}".format(assigns.codegen()))


