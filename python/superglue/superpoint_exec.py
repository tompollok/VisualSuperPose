import torch
import sys
import cv2

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

imgpath = sys.argv[1]

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
}

superpoint = SuperPoint(config).eval().to(device)

image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (640, 480))
imagef = image.astype('float32')
imagenp = torch.from_numpy(imagef/255).float()[None,None].to(device)

pred = superpoint({'image':imagenp})

kp = pred['keypoints'][0].cpu().numpy()

f = open("dect.txt", "w")
for i in range(len(kp)):
    f.write("[{},{}]\n".format(kp[i][0], kp[i][1]))
f.close()

for i in range(len(kp)):
    cv2.circle(imagef, (int(kp[i][0]), int(kp[i][1])), 1, (0,0,255))

cv2.imwrite("dect.jpg", imagef)