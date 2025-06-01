import random

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def dataAugmentation(args):

    transform = transforms.Compose([
        # transforms.RandomCrop(randCrop, padding=4),  # 在 32x32 的图像上进行随机裁剪（加 4 像素填充）
        transforms.RandomCrop(args.randomCrop),  # 在 32x32 的图像上进行随机裁剪（加 4 像素填充）
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur((3)),
    ])
    return transform


