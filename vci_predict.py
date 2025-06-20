import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# --------------------
# 5. 测试单张图像 (示例)
# --------------------
def predict(image_path):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)  # 增加batch维度
    with torch.no_grad():
        output = model(image)  # 输出: [1, 4, 62]
        pred_indices = output.argmax(dim=2)[0]  # 取预测索引
        # 反向映射索引到字符
        idx2char = {v: k for k, v in train_dataset.char2idx.items()}
        pred_text = ''.join([idx2char[idx.item()] for idx in pred_indices])
    return pred_text


# 示例: 预测单张验证码
test_img = "captcha_data/test/code_AbC1_1700000000.png"
print("Predicted:", predict(test_img))