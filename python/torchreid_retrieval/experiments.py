import os
import torch
import cv2
from PIL import Image
import numpy as np
import torchreid
import torchvision.transforms as T

root_data_dir = ""

def onnx_retrieval(model, model_name, img, onnx_path="", model_weights=""):
    model = model.cuda()

    if not os.path.isfile(onnx_path):
        if os.path.isfile(model_weights):
            model.load_state_dict(torch.load(model_weights))
        x = torch.randn((1, 3, 224, 224))
        x = x.cuda()
        torch.onnx.export(model, x, onnx_path)

    # onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    net = cv2.dnn.readNetFromONNX(onnx_path)

    images = []
    if isinstance(img, str):
        img = cv2.imread(img)
        images = [img]
    elif isinstance(img, list):
        for i in img:
            images.append(cv2.imread(i))
    elif isinstance(img, np.ndarray):
        images = [img]

    outputs = []
    for i in images:
        i = cv2.resize(i, (224, 224))
        # TODO: Normalization is missing!
        input = cv2.dnn.blobFromImage(i, 1.0, (224, 224))
        net.setInput(input)
        output = net.forward()
        outputs.append(output)
        print(output)

    equal = np.sum(np.equal(outputs[0], outputs[1]))

    """"
    counter = 0
    for  i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                for l in range(input.shape[3]):
                    print(input[i, j, k, l])
                    counter += 1
    """
    return outputs


def pytorch_model_execution(model, model_name, model_weights=""):
    model = model.cuda()
    model.eval()
    torchreid.utils.load_pretrained_weights(model, model_weights)

    '''
    mat = torch.tensor(img)
    mat = mat.permute((2, 0, 1))
    mat = mat.unsqueeze(0)
    mat = mat.float()
    mat = mat.cuda()
    '''

    # Build transform functions
    img = Image.open(root_data_dir + "train/0/0/0/000a0aee5e90cbaf.jpg").convert('RGB')
    # img = cv2.imread("/media/johanna/Volume/Studium/Semester_4/Bildauswertung_und_-fusion/GoogleLandmarksDataset
    # /fbow_images/0/000ae6116e30f972.jpg")
    transforms = []
    transforms += [T.Resize((224, 224))]
    transforms += [T.ToTensor()]
    pixel_mean = [0.485, 0.456, 0.406],
    pixel_std = [0.229, 0.224, 0.225]
    transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    preprocess = T.Compose(transforms)
    # img = T.ToPILImage(img)
    img = preprocess(img)
    img = img.unsqueeze(0).cuda()

    output = model(img)
    print("input pytorch model", img)
    print("output pytorch model", output)

    # img = cv2.imread(root_data_dir + "train/0/0/0/000a0aee5e90cbaf.jpg")
    img = Image.open(root_data_dir + "train/0/0/0/000a0aee5e90cbaf.jpg").convert('RGB')

    transforms = []
    # transforms += [T.ToPILImage()]
    transforms += [T.Resize((224, 224))]
    transforms += [T.ToTensor()]
    pixel_mean = [0.485, 0.456, 0.406],
    pixel_std = [0.229, 0.224, 0.225]
    transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    preprocess = T.Compose(transforms)
    # img = T.ToPILImage(img)
    img = preprocess(img)
    img = img.unsqueeze(0).cuda()

    output = model(img)
    print("input pytorch model", img)
    print("output pytorch model", output)

    '''
    img = cv2.imread(root_data_dir + "train/0/0/0/000a0aee5e90cbaf.jpg")
    mat = torch.tensor(img)
    mat = mat.permute((2, 0, 1))
    mat = mat.unsqueeze(0)
    mat = mat.float()
    mat = mat.cuda()
    output = model(mat)
    print("input pytorch model", mat)
    print("output pytorch model", output)

    mat = torch.zeros(1, 3, 224, 224)
    mat = mat.cuda()
    output = model(mat)
    print("input pytorch model", mat)
    print("output pytorch model", output)
    '''

    return output


def main():
    """
    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=1,
        pretrained=True
    )

    imgs = [root_data_dir + "train/0/0/0/000a0aee5e90cbaf.jpg",
            "/media/johanna/Volume/colmap_images/Gendarmenmarkt/home/wilsonkl/projects/SfM_Init/dataset_images/Gendarmenmarkt/11648097_826b97c4f6_o.jpg"]

    model_onnx = "/home/johanna/Documents/Bildauswertung/ppbaf-localization/python/torchreid_retrieval/log/resnet50/model/model-50.onnx"
    model_weights = "/home/johanna/Documents/Bildauswertung/cnn-models/resnet50_150k/model.pth.tar-50"
    #model = train_model(model_name=model_name)
    onnx_retrieval(model, model_name, img=imgs, onnx_path=model_onnx)

    #pytorch_model_execution(model, model_name, "/home/johanna/Documents/Bildauswertung/cnn-models/resnet50_150k/model.pth.tar-50")

    #pytorch_model_execution(model, img, model_name, "./log/resnet50/model/model.pth.tar-10")
    """


if __name__ == "__main__":
    main()
