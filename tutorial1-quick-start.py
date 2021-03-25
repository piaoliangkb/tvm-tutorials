import numpy as np
from numpy.lib.npyio import load

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
import tvm.testing

# Define a neuralnetwork with relay python frontend.
# User pre-defined resnet-18 network in Relay.
# Assume we will do inference on our device and the
# batch size is set to 1.
# Input iamges are RGB color images of size 244 * 244.
# Call tvm.relay.TupleWraper.astext() to show the network
# sturcture.

batch_size = 1
num_class = 1000
image_shape = (3, 244, 244)
data_shape = (batch_size,) + image_shape  # (1, 3, 244, 244)
out_shape = (batch_size, num_class)  # (1, 1000)

mod, params = relay.testing.resnet.get_workload(
    number_layours=18, batch_size=batch_size, image_shape=image_shape
)

# Show network structure
# print(mod.astext(show_meta_data=False))

# Compile this model.
# User can specify the optimization level of the compilation
# from 0 to 3, which includes operator fusion, pre-computation,
# layout transformation.
# relay.build() method returns three components:
#   - execution graph in json format
#   - TVM module library for this graph on target hardware
#   - parameter blobs of the model
# Graph-level optimization was done by Relay while tensor-level
# optimization was done by TVM. Resulting an optimized runtime module.
# relay.build() method first does a number of graph-level optmizations.
# Then register the operators (the nodes of the optimized graph) to TVM
# implementations to generate a tvm.module.
# To generate the module library, TVM will first transfer the high-level
# ID into lower intrinsic IR of the specified target backend. Then the
# machine code will be generated as the module library.
optimization_level = 3
# There will be error here using llvm
target = "llvm"
with tvm.transform.PassContext(opt_level=optimization_level):
    lib = relay.build(mod, target, target_host=target, params=params)
    print(type(lib))

# Run the generated library
# Create graph runtime and run the module on Raspberry

# Create random input
ctx = tvm.context(target, 0)  # or tvm.gpu()? what is the difference
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# Create module
module = graph_runtime.GraphModule(lib["default"](ctx))
# Set input and parameters
module.set_input("data", data)
# Run
module.run()
# Get output
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()

print(out.flatten()[0:10])

# save the graph, lib, params into sepatate files
from tvm.contrib import utils
temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
print(temp.listdir())

# load the module back
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.GraphModule(loaded_lib["default"](ctx))
# run forward execution of the graph
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()

print(out_deploy.flatten()[0:10])

# check whether the output from deployed module is consistent with origin one
tvm.testing.assert_allclose(out_deploy, out, atol=1e-7)
