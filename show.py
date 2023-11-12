


# #------------------------------------------------------------#
# 可视化Detr方法：
# spatial attention weight : (cq + oq)*pk
# combined attention weight: (cq + oq)*(memory + pk)
# 其中:
#     pk:原始特征图的位置编码;
#     oq:训练好的object queries
#     cq:decoder最后一层self-attn中的输出query
#     memory:encoder的输出
# #------------------------------------------------------------#
# 在此基础上只要稍微修改便可可视化ConditionalDetr的Fig1特征图
# #------------------------------------------------------------#
# 代码参考自:https://github.com/facebookresearch/detr/tree/colab
# #------------------------------------------------------------#

import math
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.nn.functional import dropout,linear,softmax

from hubconf import detr_resnet50
from util.misc import nested_tensor_from_tensor_list


torch.set_grad_enabled(False)


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

# COCO classes
CLASSES = [
    'background','Lighter','Knife','Wrench','Pliers','Scissors','Powerbank','Axe','Gun','Hammer','Handcuff','Liquid','Saw','Stick','Umbrella'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def plot_results(pil_img, prob, boxes):
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
    plt.show()

# 加载线上的模型
model = detr_resnet50(False, 15).cuda();
state_dict = torch.load("outputs/best_weight.pth", map_location='cpu')
model.load_state_dict(state_dict["model"])
model.eval()

im = Image.open(r'data/Xray_V/JPEGImages/00003960.jpg')
im1 = Image.open(r'data/Xray_H/JPEGImages/00003960_2.jpg')

# img_path = '/home/wujian/000000039769.jpg'
# im = Image.open(img_path)
new_size = im.size
im1 = im1.resize(new_size)

# mean-std normalize the input image (batch-size: 1)
anImg = transform(im).cuda()
anImg1 = transform(im1).cuda()
data = nested_tensor_from_tensor_list([anImg])[1]
data1 = nested_tensor_from_tensor_list([anImg1])[1]
# data = nested_tensor_from_tensor_list([anImg])

# propagate through the model
outputs = model(data,data1)

probas = outputs['pred_logits'].softmax(-1)[0,:,:-1]
keep = probas.max(-1).values > 0.9

bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,keep], im.size).cuda()
# plot_results(im, probas[keep], bboxes_scaled)

conv_features, enc_attn_weights, dec_attn_weights = [], [], []

hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        lambda self, input, output: enc_attn_weights.append(output[1])
    ),
    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
]

# propagate through the model
outputs = model(data,data1)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
enc_attn_weights = enc_attn_weights[0]
dec_attn_weights = dec_attn_weights[0]

h, w = conv_features['0'].tensors.shape[-2:]

# fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 15))

# colors = COLORS * 100
# print(bboxes_scaled)
# for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
#     ax = ax_i
#     ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu())
#     ax.axis('off')
#     ax.set_title(f'query id: {idx.item()}')
# fig.tight_layout()
# plt.show()

f_map = conv_features['0']
print('Encoder attention:', enc_attn_weights[0].shape)
print('Feature map:', f_map.tensors.shape)

shape = f_map.tensors.shape[-2:]
sattn = enc_attn_weights[0].reshape(shape + shape)
print("Reshaped self-attention:", sattn.shape)

fact = 32

# let's select 4 reference points for visualization
idxs = [(620,520)]

# here we create the canvas
fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
# and we add one plot per reference point
gs = fig.add_gridspec(2, 4)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
]

# for each one of the reference points, let's plot the self-attention
# for that point
for idx_o, ax in zip(idxs, axs):
    idx = (idx_o[0] // fact, idx_o[1] // fact)
    ax.imshow(sattn[..., idx[0], idx[1]].cpu(), cmap='cividis', interpolation='nearest')
    ax.axis('off')
    ax.set_title(f'self-attention{idx_o}')

# and now let's add the central image, with the reference points as red circles
fcenter_ax = fig.add_subplot(gs[:, 1:-1])
fcenter_ax.imshow(im)

for (y, x) in idxs:
    scale = im.height / data.tensors.shape[-2]
    x = ((x // fact) + 0.5) * fact
    y = ((y // fact) + 0.5) * fact
    fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    fcenter_ax.axis('off')
plt.show()
