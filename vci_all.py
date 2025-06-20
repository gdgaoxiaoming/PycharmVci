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


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaModel().to(device)

# --------------------
# 3. 训练与验证代码
# --------------------
# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()  # 每字符独立计算交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)


def train(epoch):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # 输出: [B, 4, 62]
        # 计算4个字符的损失总和
        loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(4))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")


def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取每字符预测概率最大值
            preds = outputs.argmax(dim=2)  # 形状: [B, 4]
            # 检查4个位置是否全部正确
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy


# --------------------
# 4. 训练循环与模型保存
# --------------------
best_acc = 0.0
for epoch in range(20):
    train(epoch)
    acc = validate()
    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model with accuracy: {acc:.2f}%")


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
# test_img = "captcha_data/test/code_AbC1_1700000000.png"
# print("Predicted:", predict(test_img))