import onnx
import onnxruntime
import numpy as np
from tvm.contrib.download import download_testdata
from scipy.special import softmax
import time

onnx_model = onnx.load("resnet50-v2-7.onnx")
onnx.checker.check_model(onnx_model)

session = onnxruntime.InferenceSession("resnet50-v2-7.onnx")

# Load from npz data.
# np.savez("imagenet_cat", data=img_data)
inputs = np.load("imagenet_cat.npz")["data"]
# Get output
print(type(inputs))
st = time.time()
outputs = session.run([], input_feed={"data": inputs})
print(f"Prediction time: {time.time()-st}")

# Post-process this output
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

scores = softmax(outputs)
print(type(scores))
print(scores.shape)  # (1, 1000)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]

for rank in ranks[0:5]:
    print(f"class={labels[rank]} with probability={scores[rank]}") 