import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# 显示图片
def show_pics(pics):
    plt.figure(figsize=(3 * len(pics), 3), dpi=80)
    for i, x in enumerate(pics):
        img = x[0].detach().cpu().numpy().transpose(1,2,0)
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)  # 修复 1：防止越界报错
        plt.subplot(1, len(pics), i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 打开单张图片
def open_pic(path):
    img = Image.open(path).resize((256,256), Image.BILINEAR).convert('RGB')
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img.transpose(2,0,1)
    return torch.from_numpy(img).unsqueeze(0)

# 保存图片
def save_pics(pics, name, path='./output/pics/'):
    tensors = [x[0].detach().cpu().numpy() for x in pics]
    img = np.concatenate(tensors, axis=2)
    img = img.transpose(1,2,0)
    img = (img + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(f"{path}/{name}.jpg")

# 图片缓存池，CycleGAN 专用
class ImagePool:
    def __init__(self, size=50):
        self.pool = []
        self.size = size

    def query(self, x):
        res = []
        for img in x:
            img = img.unsqueeze(0)
            if len(self.pool) < self.size:
                self.pool.append(img)
                res.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.size)
                    res.append(self.pool[idx])
                    self.pool[idx] = img
                else:
                    res.append(img)
        return torch.cat(res)
