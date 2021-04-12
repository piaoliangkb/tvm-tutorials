import time

import numpy as np
import onnx
import onnxruntime
from scipy.special import softmax
from tvm.contrib.download import download_testdata

# ref:
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb
# https://pytorch.org/docs/stable/onnx.html
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

model_path = "resnet152/resnet152-v2-7.onnx"

onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

session = onnxruntime.InferenceSession(model_path)

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
    labels = [line.rstrip() for line in f]

scores = softmax(outputs)
print(type(scores))
print(scores.shape)  # (1, 1000)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]

for rank in ranks[0:5]:
    print(f"class={labels[rank]} with probability={scores[rank]}")
