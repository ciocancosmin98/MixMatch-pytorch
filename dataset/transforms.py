import numpy as np
import os

import torchvision.transforms as transforms
#from torchvision.transforms import ToTensor
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import InterpolationMode
import torch

def load_from_list(transform_list):
    result_list = []
    tensor_list = []
    """
    pil_list = []
    pil_list.append(ToPILImage())
    """

    for transform in transform_list:
        if transform['name'] == 'RandomPadandCrop':
            result_list.append(RandomPadandCrop())
        elif transform['name'] == 'RandomFlip':
            result_list.append(RandomFlip())
        elif transform['name'] == 'RandomPadRotate':
            tensor_list.append(
                RandomRotate(
                    float(transform['degrees']),
                    transform['interpolation']
                )
            )

    """
    elif transform['name'] == 'ColorJitter':
        tensor_list.append(RandomColorJitter(
            float(transform['brightness']),
            float(transform['contrast']),
            float(transform['saturation']),
            float(transform['hue'])
        ))
    """

    result_list.append(ToTensorFromNumpy())
    result_list.extend(tensor_list)

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

class RandomPadandCrop():
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

"""
class RandomColorJitter():
    def __init__(self, brightness, contrast, saturation, hue):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    def __call__(self, x):
        x = self.transform(x)
        return x
"""
class RandomRotate():
    def __init__(self, degrees, interpolation):
        assert isinstance(degrees, float)

        if interpolation == "NEAREST":
            interpolation = InterpolationMode.NEAREST
        elif interpolation == "BILINEAR":
            interpolation = InterpolationMode.BILINEAR
        else:
            raise TypeError(f"Transform interpolation {interpolation} not supported.")
        self.rotate = RandomAffine(degrees,
                interpolation=interpolation)
        self.pad = None

    def __call__(self, x):
        old_h, old_w = x.shape[1:]

        if self.pad is None:
            self.pad_size = x.shape[1] // 4
            self.hp = self.pad_size // 2
            self.pad = transforms.Pad(self.pad_size, padding_mode='reflect')

        x = self.pad(x)

        h, w = x.shape[1:]

        x = self.rotate(x)

        top = np.random.randint(self.hp, h - old_h - self.hp)
        left = np.random.randint(self.hp, w - old_w - self.hp)

        x = x[:, top: top + old_h, left: left + old_w]
        return x

class RandomFlip():
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise():
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensorFromNumpy():
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 