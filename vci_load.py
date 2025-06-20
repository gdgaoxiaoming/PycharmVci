import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------
# 1. 数据加载 (适配命名格式)
# --------------------
class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "code_*.png"))
        self.transform = transform
        # 62 类字符映射字典（0-9, A-Z, a-z）
        self.char2idx = {c: i for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 解析文件名格式：code_验证码内容_时间戳.png
        label_str = os.path.basename(img_path).split("_")[1]  # 提取验证码内容
        # 加载灰度图并转换张量
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        # 将4位验证码转为4个独立标签索引
        labels = [self.char2idx[char] for char in label_str]
        return image, torch.tensor(labels, dtype=torch.long)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 120)),  # 统一输入尺寸
    transforms.ToTensor(),  # 转张量
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
])

# 创建数据集与加载器
train_dataset = CaptchaDataset(root_dir="data/train", transform=transform)
test_dataset = CaptchaDataset(root_dir="data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
