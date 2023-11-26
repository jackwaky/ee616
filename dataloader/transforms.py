import torch
import numpy as np
from torch import nn
from torchvision.transforms import transforms

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )


    def __call__(self, img):

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()


        return img

def get_gaussian_blur(size=224, s=1, prob=1):
    assert(s<=1.2 and s>=0.8)
    assert(prob<=1 and prob>=0.5)
    gaussian_blur = transforms.Compose([
        GaussianBlur(kernel_size=int(0.1*s * size))
    ])
    gaussian_blur = transforms.RandomApply([gaussian_blur], p=prob)
    return gaussian_blur

def get_random_crop(size=224, s=1, prob=1):
    assert(s<=2 and s>=1)
    assert(prob<=1 and prob>=0.5)
    random_crop = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2*s, 1.0)),
        transforms.RandomHorizontalFlip()
    ])
    random_crop = transforms.RandomApply([random_crop], p=prob)
    return random_crop

def get_color_distortion(s=1, prob=1):
    assert(s<=2 and s>=1)
    assert(prob<=1 and prob>=0.5)
    color_jitter = transforms.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    color_distort = transforms.RandomApply([color_distort], p=prob)
    return color_distort