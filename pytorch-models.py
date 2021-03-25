import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime

import torch
import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image


# Global log func
def logs(info: str):
    print(f"=================: {info}")


# Load pre-trained pytorch model

# equels to: model = torchvision.models.resnet18
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
# What does model.eval() do in pytorch?
# https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()  # get PyTorch graph representation
# print(type(scripted_model))
# print(scripted_model)

# Load test image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Process image and convert to tensor

# Compose several transforms together
my_preprocess = transforms.Compose(
    [
        # Resize input image to the given size
        transforms.Resize(256),
        # Crops the image at the center
        transforms.CenterCrop(224),
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# before process
print(img)
# after process
print(type(img))
img = my_preprocess(img)
print(type(img))
print(img.shape) 
img = np.expand_dims(img, 0)
print(type(img))
print(img.shape)
# In conclusion: PIL.Image -> torch.Tensor (3, 224, 224) -> numpy.ndarray (1, 3, 224, 224)
# Finish loading test image

# Convert Pytorch graph to Relay graph
logs("Convert Pytorch graph to TVM Relay graph")
input_name = "some_input"
shape_list = [(input_name, img.shape)]
print(shape_list)
# Get TVM Relay Module (IR Module) and TVM runtime param
# Input parameters to the graph that do not change during inference time. Used for constant folding.
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Relay build
logs("Compile traph to llvm target with given input specification")
target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)

# Deploy the compiled module on target
logs("Use compiled module to test input")
dtype = "float32"
print(lib["default"])
print(type(lib["default"]))
print(lib["default"](ctx))
# Get graph module
m = graph_runtime.GraphModule(lib["default"](ctx))
# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# Run forward execution of the graph
m.run()
# Get output
# Use Compiled TVM module and pytorch image to get output
tvm_output = m.get_output(0)
# print(tvm_output)
print("module output type: ", type(tvm_output))

# Lookup synset name
logs("Lookup prediction top 1 index in 1000 class synset")
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [item.strip() for item in synsets]
# print(synsets)
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}
# print(key_to_classname)

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]
# print(class_id_to_key)

# Get top-1 result of TVM
logs("Get top-1 result of TVM")
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison

# Disable gradient calculation
# Use pytorch model to predict image
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    print("Get torch image type: ", type(torch_img))
    output = model(torch_img)
    print("Use torch image as torch model input, get output type: ", type(output))

    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
