# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection_main(torchvision.datasets.CocoDetection):
    def __init__(self, name, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection_main, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.name = name

    def __getitem__(self, idx):
        img, target = super(CocoDetection_main, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        return img, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_info, transforms, return_masks, mode):

        img_folder_main, ann_file_main = img_info["train_main"]
        self.CoCo_main = CocoDetection_main("main", img_folder_main, ann_file_main,
                                            transforms=transforms, return_masks=return_masks)
        if mode == "train":
            img_folder_assist, ann_file_assist = img_info["train_assist"]
            self.CoCo_assist = CocoDetection_main("assist", img_folder_assist, ann_file_assist,
                                                  transforms=transforms, return_masks=return_masks)
            assert len(self.CoCo_main) == len(self.CoCo_assist)
        if mode == "val":
            img_folder_assist, ann_file_assist = img_info["train_assist"]
            self.CoCo_assist = CocoDetection_main("assist", img_folder_assist, ann_file_assist,
                                                  transforms=transforms, return_masks=return_masks)
            assert len(self.CoCo_main) == len(self.CoCo_assist)
        self.mode = mode
        self._transforms = transforms

    def __getitem__(self, idx):
        img_main, target_main = self.CoCo_main[idx]
        if (self.CoCo_main.ids[idx] in self.CoCo_main.coco.imgs.keys()):
            file_name_main = self.CoCo_main.coco.imgs[self.CoCo_main.ids[idx]]["file_name"]

        if self.mode == "train":
            img_assist, target_assist = self.CoCo_assist[idx]
            if (self.CoCo_main.ids[idx] in self.CoCo_assist.coco.imgs.keys()):
                file_name_assist = self.CoCo_assist.coco.imgs[self.CoCo_main.ids[idx]]["file_name"]
                assert file_name_main[:-4] == file_name_assist[:-6]
            # if "3964" in file_name_main:
            #     print("3964")
            if self._transforms is not None:
                img_main, target_main, img_assist, target_assist = self._transforms(img_main, target_main, img_assist,
                                                                                    target_assist)

            return img_main, target_main, img_assist, target_assist, file_name_main, file_name_assist

        if self.mode == "val":
            img_assist, target_assist = self.CoCo_assist[idx]
            if (self.CoCo_main.ids[idx] in self.CoCo_assist.coco.imgs.keys()):
                file_name_assist = self.CoCo_assist.coco.imgs[self.CoCo_main.ids[idx]]["file_name"]
                assert file_name_main[:-4] == file_name_assist[:-6]
            # if "3964" in file_name_main:
            #     print("3964")
            if self._transforms is not None:
                img_main, target_main, img_assist, target_assist = self._transforms(img_main, target_main, img_assist,
                                                                                    target_assist)

            return img_main, target_main, img_assist, target_assist, file_name_main, file_name_assist
        # else:
        #     if self._transforms is not None:
        #         img_main, target_main,_,_ = self._transforms(img_main, target_main)
        #     return img_main, target_main, file_name_main

    def __len__(self):
        return len(self.CoCo_main)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": {"train_main": (root / "Xray_V/JPEGImages", root / "Xray_V/annotations" / f'{mode}_train2017.json'),
                  "train_assist": (root / "Xray_H/JPEGImages", root / "Xray_H/annotations" / f'{mode}_train2017.json')},
        "val": {"train_main": (root / "Xray_V/JPEGImages", root / "Xray_V/annotations" / f'{mode}_val2017.json'),
                "train_assist": (root / "Xray_H/JPEGImages", root / "Xray_H/annotations" / f'{mode}_val2017.json')}

        ############----------------------俯视图只训练，不参与验证，所以不用载入俯视图的验证集，后期修改---------------############
    }

    img_info = PATHS[image_set]
    dataset = CocoDetection(img_info=img_info,
                            transforms=make_coco_transforms(image_set), return_masks=args.masks, mode=image_set)
    return dataset
