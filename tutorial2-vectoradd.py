from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt_host = "llvm"
tgt = "c"

# n representes the shape
n = te.var("n")
# two placeholder Tensors, given shape (n, )
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
# result Tensor C with a compute operation
# takes
#   - the shape of the tensor
#   - a lambda function describing the computation rule
#   for each position of the tensor
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))

# create schedule
s = te.create_schedule(C.op)

func_add = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="vectorAdd")

print(func_add.get_source())

ctx = tvm.context(tgt, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, C.dtype), ctx)
func_add(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

print(a)
print(b)
print(c)

# save compiled model
from tvm.contrib import cc, utils

temp = utils.tempdir()
func_add.save(temp.relpath("func_add.o"))

print(func_add)