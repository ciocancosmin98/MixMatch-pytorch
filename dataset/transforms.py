import numpy as np
import os

import torchvision.transforms as transforms
import torch

def load_from_list(transform_list):
    result_list = []

    for transform in transform_list:
        if transform['name'] == 'RandomPadandCrop':
            result_list.append(RandomPadandCrop())
        elif transform['name'] == 'RandomFlip':
            result_list.append(RandomFlip())

    result_list.append(ToTensor())

    return transforms.Compose(result_list)

def load_transforms(transform_name):
    transform_path = os.path.join('transforms', transform_name)

    """
    if transform_path is None or not os.path.exists(transform_path):
        # use default transforms
        transform_train = transforms.Compose([
            RandomPadandCrop(),
            RandomFlip(),
            ToTensor(),
        ])

        transform_val = transforms.Compose([
            ToTensor(),
        ])
    """
    if transform_path is None or not os.path.exists(transform_path):
        raise Exception("Cannot find transforms file.")
        
    import json

    with open(transform_path) as f:
        data = json.load(f)

    transform_train = load_from_list(data['transforms_train'])
    transform_val   = load_from_list(data['transforms_val'])
        
    return transform_train, transform_val
        
    
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

        
def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    def __init__(self):
        pass

    def __call__(self, x):
        old_h, old_w = x.shape[1:]

        x = pad(x, old_h // 8)

        h, w = x.shape[1:]

        top = np.random.randint(0, h - old_h)
        left = np.random.randint(0, w - old_w)

        x = x[:, top: top + old_h, left: left + old_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 