import torchreid
import torch.onnx
import torch
import onnx
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

# to change
# source = "D:/Uni/Projektpraktikum/cnn-models/resnet50_150k/model.pth.tar-10"
source = "/home/johanna/Documents/Bildauswertung/cnn-models/resnet50_150k/model.pth.tar-50"
destination = "./log/resnet50/model/model-50.onnx"
# dest = "D:/Uni/Projektpraktikum/onnx-models/resnet50_150k/model-10.onnx"
modelName = "resnet50"
numClasses = 20152
# to change end
model = torchreid.models.build_model(
    name=modelName,
    num_classes=20152,
    loss="softmax",
    pretrained=True
)
x = torch.randn((1, 3, 224, 224))
x = x.cuda()
model = model.cuda()
model.eval()
torchreid.utils.load_pretrained_weights(model, source)
traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("traced_retrieval_model.pt")
