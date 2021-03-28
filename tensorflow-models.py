import tvm
from tvm import te
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime

import numpy as np
from PIL import Image

import os.path
import time

import tensorflow as tf
# Import tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Deploying tensorflow models with TVM

# Global log func
def logs(info: str):
    print(f"=================: {info}")

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf


# Model related file
repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"

# Test image
img_name = "elephant-299.jpg"
image_url = os.path.join(repo_base, img_name)

# Get model url
model_name = "classify_image_graph_def-with_shapes.pb"
model_url = os.path.join(repo_base, model_name)

# Image label map
map_proto = "imagenet_2012_challenge_label_map_proto.pbtxt"
map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
label_map = "imagenet_synset_to_human_label_map.txt"
label_map_url = os.path.join(repo_base, label_map)

# Download files
img_path = download_testdata(image_url, img_name, module="data")
model_path = download_testdata(model_url, model_name, module=["tf", "InceptionV1"])
print(model_path)
map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
label_path = download_testdata(label_map_url, label_map, module="data")

# Build model for cpu
target = "llvm"
target_host = "llvm"
layout = None
ctx = tvm.cpu(0)

# Create tensorflow graph defined in protobuf file
logs("Create tensorflow graph from protobuf file")
with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    # graph is a list of operation or tensor objects
    graph = tf.import_graph_def(graph_def, name="")
    # import the graph definition into default graph
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    print("graph_def type: ", type(graph_def))
    # add shapes to the graph
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")

image = Image.open(img_path).resize((299, 299))
x = np.array(image)

# Import the graph to relay
logs("Import tensorflow graph to relay")
shape_dict = {"DecodeJpeg/contents": x.shape}
dtype_dict = {"DecodeJpeg/contents": "uint8"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

# Compile the graph to llvm target with given input specification
logs("Compile TVM IR for tensorflow graph")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

# Deploying the compiled module on target
dtype = "uint8"
m = graph_runtime.GraphModule(lib["default"](ctx))
# set inputs
m.set_input("DecodeJpeg/contents", tvm.nd.array(x.astype(dtype)))
# execute
st = time.time()
m.run()
print("TVM prediction time = ", time.time() - st)
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), "float32"))

# Process output
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path, uid_lookup_path=label_path)

# Print top 5 predictions from TVM output.
print("===== TVM RESULTS =======")
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print("%s (score = %.5f)" % (human_string, score))


# Run tensorflow inference
def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)


def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf_compat_v1.gfile.Exists(image):
        tf.logging.fatal("File does not exist %s", image)
    image_data = tf_compat_v1.gfile.GFile(image, "rb").read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf_compat_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
        st = time.time()
        predictions = sess.run(softmax_tensor, {"DecodeJpeg/contents:0": image_data})
        print("Tensorflow prediction time = ", time.time() - st)

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = tf_testing.NodeLookup(
            label_lookup_path=map_proto_path, uid_lookup_path=label_path
        )

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print("%s (score = %.5f)" % (human_string, score))


run_inference_on_image(img_path)
