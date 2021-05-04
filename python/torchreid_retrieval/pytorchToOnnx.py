import torchreid
import torch.onnx
import onnx
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

# adapt source and destination paths to your system
source = "/media/johanna/Volume/Studium/Semester_4/Bildauswertung_und_-fusion/cnns_whole_dataset/model.pth.tar-80"
destination = "/media/johanna/Volume/Studium/Semester_4/Bildauswertung_und_-fusion/cnns_whole_dataset/model-80.onnx"
modelName = "resnet50"
numClasses = 70566 #for whole train clean dataset
#numClasses = 20152 #for small train clean dataset
# to change end
model = torchreid.models.build_model(
    name=modelName,
    num_classes=numClasses,
    loss="softmax",
    pretrained=True
)
x = torch.randn((1, 3, 224, 224))
x = x.cuda()
model = model.cuda()
model.eval()
torchreid.utils.load_pretrained_weights(model, source)
torch.onnx.export(model, x, destination)
