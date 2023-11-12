from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

files_path_H = list(Path(r"data/Xray_2/JPEGImage").rglob("*.jpg"))
files_path_V = list(Path(r"data/Xray_1/JPEGImage").rglob("*.jpg"))

for i in range(10):
    plt.figure(figsize=(15, 7))
    print(files_path_H[i])
    assert str(files_path_H[i]).split("\\")[-1].split("_")[0] == str(files_path_V[i]).split("\\")[-1].split("_")[0]
    img_H = Image.open(files_path_H[i])
    plt.title(str(files_path_H[i]).split("\\")[-1])
    img_V = Image.open(files_path_V[i])
    img_V = F.resize(img_V, (img_V.size[1], img_H.size[0]))
    img_H_np = np.asarray(img_H)
    img_V_np = np.asarray(img_V)
    plt.title(str(files_path_V[i]).split("\\")[-1])

    plt.imshow(img_V)
    plt.show()
