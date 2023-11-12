# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate
import matplotlib.pyplot as plt


def crop(image_main, target_main, image_assist, target_assist, region):
    region_main = (region[0], 0, image_main.size[1], region[3])
    region_assist = (region[0], 0, image_assist.size[1], region[3])
    cropped_image_main = F.crop(image_main, *region_main)
    cropped_image_assist = F.crop(image_assist, *region_assist)

    target_main = target_main.copy()
    target_assist = target_assist.copy()
    i_main, j_main, h_main, w_main = region_main
    i_assist, j_assist, h_assist, w_assist = region_assist

    # should we do something wrt the original size?
    target_main["size"] = torch.tensor([h_main, w_main])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target_main:
        boxes = target_main["boxes"]
        max_size = torch.as_tensor([w_main, h_main], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j_main, i_main, j_main, i_main])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target_main["boxes"] = cropped_boxes.reshape(-1, 4)
        target_main["area"] = area
        fields.append("boxes")

    if "masks" in target_main:
        # FIXME should we update the area here if there are no boxes?
        target_main['masks'] = target_main['masks'][:, i_main:i_main + h_main, j_main:j_main + w_main]
        fields.append("masks")

    if "masks" in target_assist:  # fixed
        target_assist['masks'] = target_assist['masks'][:, i_assist:i_assist + h_assist,
                                 j_assist:j_assist + w_assist]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target_main or "masks" in target_main:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target_main:
            cropped_boxes = target_main['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target_main['masks'].flatten(1).any(1)

        for field in fields:
            target_main[field] = target_main[field][keep]

    return cropped_image_main, target_main, cropped_image_assist, target_assist

def hflip(image, target):
    if image is None and target is None:
        return None, None
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image_main, target_main, image_assist, target_assist, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # 以主视图为标准，在水平轴上进行resize后，进行对齐
    size = get_size(image_main.size, size, max_size)
    rescaled_image_main = F.resize(image_main, size)
    rescaled_image_assist = F.resize(image_assist, size)
    target_main = target_main.copy()
    target_assist = target_assist.copy()
    if "boxes" in target_main:
        # scale bounding boxes
        boxes = target_main["boxes"]
        scaled_boxes = boxes / torch.tensor(
            [image_main.size[1], image_main.size[0], image_main.size[1], image_main.size[0]],
            dtype=torch.float32,
        )
        scaled_boxes *= torch.tensor([rescaled_image_main.size[1], rescaled_image_main.size[0], rescaled_image_main.size[1], rescaled_image_main.size[0]], dtype=torch.float32)
        target_main["boxes"] = scaled_boxes

    if "masks" in target_main:
        # FIXME need to update mask scaling
        pass

    if "boxes" in target_assist:
        # scale bounding boxes
        boxes = target_assist["boxes"]
        scaled_boxes = boxes / torch.tensor(
            [image_assist.size[1], image_assist.size[0], image_assist.size[1], image_assist.size[0]],
            dtype=torch.float32,
        )
        scaled_boxes *= torch.tensor([rescaled_image_assist.size[1], rescaled_image_assist.size[0], rescaled_image_assist.size[1], rescaled_image_assist.size[0]], dtype=torch.float32)
        target_assist["boxes"] = scaled_boxes

    if "masks" in target_assist:
        # FIXME need to update mask scaling
        pass

    return rescaled_image_main, target_main, rescaled_image_assist, target_assist

    # plt.close()
    # plt.imshow(rescaled_image_main)
    # plt.savefig("./1.png")
    # plt.imshow(rescaled_image_assist)
    # plt.savefig("./2.png")

    if target_main is None:
        return rescaled_image_main, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image_main.size, image_main.size))
    ratio_width, ratio_height = ratios

    target_main = target_main.copy()
    if "boxes" in target_main:
        boxes = target_main["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target_main["boxes"] = scaled_boxes

    if "area" in target_main:
        area = target_main["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target_main["area"] = scaled_area

    h, w = size
    target_main["size"] = torch.tensor([h, w])

    if "masks" in target_main:
        target_main['masks'] = interpolate(
            target_main['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image_main, target_main, rescaled_image_assist, target_assist


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_main: PIL.Image.Image, target_main: dict, img_assist: PIL.Image.Image, target_assist: dict):
        w = random.randint(self.min_size, min(img_main.width, self.max_size))
        h = random.randint(self.min_size, min(img_main.height, self.max_size))
        region = T.RandomCrop.get_params(img_main, [h, w])
        return crop(img_main, target_main, img_assist, target_assist, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_main, target_main, img_assist, target_assist):
        if random.random() < self.p:
            img_main, target_main = hflip(img_main, target_main)
            img_assist, target_assist = hflip(img_assist, target_assist)
        return img_main, target_main, img_assist, target_assist


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image_main, target_main, image_assist=None, target_assist=None):
        size = random.choice(self.sizes)
        return resize(image_main, target_main, image_assist, target_assist, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image_main, target_main, image_assist, target_assist):
        if random.random() < self.p:
            return self.transforms1(image_main, target_main, image_assist, target_assist)
        return self.transforms2(image_main, target_main, image_assist, target_assist)


class ToTensor(object):
    def __call__(self, img_main, target_main, img_assist, target_assist):
        if img_assist is None:
            return F.to_tensor(img_main), target_main, None, None
        else:
            return F.to_tensor(img_main), target_main, F.to_tensor(img_assist), target_assist


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_main, target_main, image_assist=None, target_assist=None):
        image_main = F.normalize(image_main, mean=self.mean, std=self.std)
        if image_assist is not None:
            image_assist = F.normalize(image_assist, mean=self.mean, std=self.std)
        if target_main is None:
            return image_main, None
        target_main = target_main.copy()
        h, w = image_main.shape[-2:]
        if "boxes" in target_main:
            boxes = target_main["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target_main["boxes"] = boxes
        return image_main, target_main, image_assist, target_assist


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_main, target_main, image_assist=None, target_assist=None):
        for t in self.transforms:
            image_main, target_main, image_assist, target_assist = t(image_main, target_main, image_assist, target_assist)

        return image_main, target_main, image_assist, target_assist

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
