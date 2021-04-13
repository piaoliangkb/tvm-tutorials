from tvm.contrib.download import download, download_testdata
from PIL import Image

import numpy as np


# Pre-processing
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.jpg", module="data")

# Resize to 244*244
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")  # (224, 224, 3)
# ONNX expects NCHW input, convert the array
img_data = np.transpose(img_data, (2, 0, 1))  # (3, 224, 224)

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")

for i in range(img_data.shape[0]):
    norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# Stack this numpy array on first daemon
test_img = np.stack((norm_img_data,) * 4, axis=0)  # (4, 3, 224, 224)
np.savez("imagenet_cat_batch", data=test_img)

# Add batch dimensions
img_data = np.expand_dims(norm_img_data, axis=0)  # (1, 3, 224, 224)
np.savez("imagenet_cat", data=img_data)


# Post-processing
import os.path
from scipy.special import softmax

labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        print(type(scores))
        print(scores.shape)  # (1, 1000)
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print(f"class={labels[rank]} with probability={scores[rank]}")
