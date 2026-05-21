import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PairDataset(Dataset):
    def __init__(self, root, dynamic_balance=True):
        """
        Args:
            root: 数据集根目录
            dynamic_balance: 是否启用动态欠采样（True=平衡，False=保持原始比例）
        """
        # 加载所有图片路径
        self.A = sorted([
            os.path.join(root, 'trainA', x) for x in os.listdir(f"{root}/trainA")
            if x.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')) and not x.startswith('.')
        ])
        self.B = sorted([
            os.path.join(root, 'trainB', x) for x in os.listdir(f"{root}/trainB")
            if x.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')) and not x.startswith('.')
        ])
        
        self.dynamic_balance = dynamic_balance
        
        # A 是少数类（莫奈）
        self.len_A = len(self.A)
        self.len_B_original = len(self.B)
        
        if not dynamic_balance:
            # 传统模式：取最小值
            self.len = min(self.len_A, self.len_B_original)
        else:
            #动态模式：长度 = 少数类 A 的数量
            self.len = self.len_A
        
        self.current_B_indices = None
        
    def set_epoch(self, epoch=None):
        """每个epoch开始时调用，重新采样B的索引"""
        if self.dynamic_balance:
            # 从多数类 B 中随机抽取 len_A 个索引（匹配少数类A）
            self.current_B_indices = random.sample(range(self.len_B_original), self.len_A)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # A 是少数类，直接按顺序取
        a_path = self.A[idx % len(self.A)]
        
        # B 是多数类，使用动态采样
        if self.dynamic_balance and self.current_B_indices is not None:
            b_idx = self.current_B_indices[idx]
        else:
            b_idx = idx % len(self.B)
        
        b_path = self.B[b_idx]
        
        a = self.load(a_path)
        b = self.load(b_path)
        return a, b
    
    def load(self, path):
        img = Image.open(path).resize((256,256)).convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)



class TestDataset(Dataset):
    # 增加一个参数：test_folder，用来指定读 testA 还是 testB
    def __init__(self, root, test_folder="testB"):
        # 自动拼接路径：root/testA  或  root/testB
        self.img_dir = os.path.join(root, test_folder)
        
        self.files = sorted([
            os.path.join(self.img_dir, x) for x in os.listdir(self.img_dir)
            if x.lower().endswith(('jpg', 'png', 'jpeg', 'bmp')) and not x.startswith('.')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).resize((256, 256)).convert('RGB')
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

