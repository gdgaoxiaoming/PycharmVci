import torch
import torch.nn as nn


# --------------------
# 2. 模型架构 (适配输入输出)
# --------------------
class CaptchaModel(nn.Module):
    def __init__(self, num_chars=4, num_classes=62):
        super().__init__()
        # 卷积层组 (特征提取)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [32,120] -> [16,32,120]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [16,16,60]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> [32,16,60]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32,8,30]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [64,8,30]
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> [64,4,15]
        )
        # 全连接层 -> 分拆为4个分类头
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 15, 1024),  # 输入特征维度: 64 * 4 * 15=3840
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # 4个独立输出层 (解耦多字符分类)
        self.heads = nn.ModuleList([
            nn.Linear(1024, num_classes) for _ in range(num_chars)
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return torch.stack([head(x) for head in self.heads], dim=1)  # 输出形状: [B, 4, 62]
