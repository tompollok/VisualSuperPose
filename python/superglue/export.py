import torch
import torch.onnx
import sys

from superpoint_simple import SuperPoint
#from models.superpoint import SuperPoint
from models.superglue import SuperGlue

config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
        'weights': sys.argv[1]
}

superpoint = SuperPoint(config)
superpoint.eval()

def export_superpoint():
        x = torch.randn(1, 1, 480, 640)
        #out = superpoint(x)
        torch.onnx.export(superpoint, x, "superpoint_simple.onnx", export_params=True, opset_version=10)


def export_superglue():
        config = {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }

        superglue = SuperGlue(config)
        superglue.eval()
        x1 = torch.randn(1, 1, 480, 640)
        x2 = torch.randn(1, 1, 480, 640)

        pred0 = self.superpoint({'image': x1})
        pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        
        pred1 = self.superpoint({'image': x2})
        pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        y = superpoint(x)
        torch.onnx.export(superglue, y, "superglue.onnx", export_params=True, opset_version=10)

export_superpoint()
#export_superglue()