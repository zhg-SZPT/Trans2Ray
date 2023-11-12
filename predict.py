import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import*
from util.misc import nested_tensor_from_tensor_list
torch.set_grad_enabled(False)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# COCO classes
CLASSES = [
    'background','Lighter','Knife','Wrench','Pliers','Scissors','Powerbank','Axe','Gun','Hammer','Handcuff','Liquid','Saw','Stick','Umbrella'
]  #设置一定是第一个类别为背景，然后加上要检测类别名

# colors for visualization
COLORS = [[1.000, 0.000, 0.000],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    return b

def plot_results(pil_img, prob, boxes,save_path=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    result_path = os.path.join('showplt', 'result.txt')
    with open(result_path, 'w') as f:
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            class_name = CLASSES[cl]
            confidence = p[cl]
            result_str = f'Class: {class_name}, Confidence: {confidence:.2f}, Bounding Box: ({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})\n'
            print(result_str)
            f.write(result_str)

    if save_path is not None:
        image_path = os.path.join('showplt', 'image.jpg')
    plt.savefig(image_path)
    plt.show()



def detect(im, im1, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).cuda()
    img1 = transform(im1).cuda()

    img = nested_tensor_from_tensor_list([img])[1]
    img1 = nested_tensor_from_tensor_list([img1])[1]

    model = model.cuda()
    # propagate through the model
    outputs = model(img, img1)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.00001

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def predict(im, im1, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im).cuda()
    anImg1 = transform(im1).cuda()

    main_data = nested_tensor_from_tensor_list([anImg])[1]
    assist_data = nested_tensor_from_tensor_list([anImg1])[1]

    model = model.cuda()

    # propagate through the model
    outputs = model(main_data, assist_data)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    keep = probas.max(-1).values > 0.7  #预测阈值大小
    #print(probas[keep])

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

if __name__ == "__main__":

    model = detr_resnet50(False, 15)  # 类别数+1
    state_dict = torch.load("outputs/best_weight.pth",map_location='cpu') #放入训练好的模型进行预测
    model.load_state_dict(state_dict["model"])
    model.eval()

    im = Image.open(r'data/Xray_V/JPEGImages/00003960.jpg') #需要测试的图片
    im1 = Image.open(r'data/Xray_H/JPEGImages/00003960_2.jpg')
    new_size = im.size
    im1 = im1.resize(new_size)
    scores, boxes = predict(im, im1, model, transform)

    save_path = 'showplt'
    plot_results(im, scores, boxes,save_path)







