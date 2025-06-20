import torch
import torch.nn as nn
import torch.optim as optim

from vci_model import CaptchaModel
from vci_load import *

model = CaptchaModel()
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
