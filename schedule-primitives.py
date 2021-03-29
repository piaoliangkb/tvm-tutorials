import tvm
from tvm import te
import numpy as np


# Global log func
def logs(info: str):
    print(f"=================: {info}")


n = te.var("n")
m = te.var("m")


###############################
########### split #############
###############################
# split works on single axis.

example = "Element-wise multiply in matrix"
logs(example)
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

s = te.create_schedule([C.op])
# Use tvm.lower() to transform the computation to the real callable
# function. Return a readable C like statement with argument
# "simple_mode=True"
logs(f"No split:\n{tvm.lower(s, [A, B, C], simple_mode=False)}")


# Use split to specified axis into two axes by factor.
example = "multiply vector by scalar"
logs(example)
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")
s = te.create_schedule(B.op)
logs(f"Before split:\n{tvm.lower(s, [A, B], simple_mode=True)}")
xo, xi = s[B].split(B.op.axis[0], factor=32)
logs(
    f"After split at axis[0] with factor=32:\n{tvm.lower(s, [A, B], simple_mode=True)}"
)

# Use nparts to split the axis contrary with factor
example = "B = A in vector"
logs(example)
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i], name="B")

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32)
logs(
    f"After split at axis[0] with nparts=32:\n{tvm.lower(s, [A, B], simple_mode=True)}"
)


###############################
############ tile #############
###############################
# Compared with split, tile can works over two axes.

# Execute the computation tile by tile over 2 axes.
example = "B = A in matrix"
logs(example)
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
logs(
    f"Tile over two axis with x_factor=10, y_factor=5:\n{tvm.lower(s, [A, B], simple_mode=True)}"
)


###############################
############ fuse #############
###############################
# Fuse two consecutive axes of one computation.
example = "B = A in matrix"
logs(example)
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
# first, tile for 4 axes
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
logs(f"First, tile 2 axes into 4 axes:\n{tvm.lower(s, [A, B], simple_mode=True)}")
# second, fuse
# What is floordiv, floormod?
# floordiv: 向下取整除 floordiv(5.5, 2) = 2
# floormod: 向下取整模 floormod(5.5, 2) = 1
# Example in Halide language:
# https://halide-lang.org/docs/tutorial_2lesson_05_scheduling_1_8cpp-example.html
fused = s[B].fuse(xi, yi)
logs(f"Second, fuse two inner axes into one:\n{tvm.lower(s, [A, B], simple_mode=True)}")
