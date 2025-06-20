PyTorch 验证码识别系统

!https://via.placeholder.com/800x400?text=Captcha+Recognition+Demo  
一个基于PyTorch的验证码识别系统，能够自动识别包含数字和字母的4位验证码。项目包含完整的训练流程和预测功能，准确率可达80.5%。

项目简介

本项目实现了一个端到端的验证码识别系统，主要特点包括：
自动生成训练数据集（支持自定义字符集和数量）

基于CNN的深度学习模型架构

完整的训练、验证和预测流程

支持GPU加速训练

模型保存与加载功能

目录结构

PycharmVci/
├── data/                   # 数据目录
├── train/              # 训练集验证码

└── test/               # 测试集验证码

├── best_model.pth          # 训练好的最佳模型
├── code_gen.py             # 验证码生成脚本
├── vci_all.py              # 主训练脚本（包含数据加载、模型训练和预测）
├── vci_model.py            # 模型架构定义
├── vci_predict.py          # 预测脚本
└── vci_train.py            # 训练脚本

快速开始

安装依赖

pip install torch torchvision pillow numpy

生成训练数据

python code_gen.py

默认生成1000张验证码到data/train目录

训练模型

python vci_train.py

或使用主脚本：
python vci_all.py

使用训练好的模型进行预测

from vci_predict import predict

result = predict("data/test/code_AbC1_1234567890.png")
print(f"预测结果: {result}")

模型架构

graph TD
    A[输入图像 32x120] --> B[卷积层 16@32x120]
--> C[ReLU激活]

--> D[最大池化 16@16x60]

--> E[卷积层 32@16x60]

--> F[ReLU激活]

--> G[最大池化 32@8x30]

--> H[卷积层 64@8x30]

--> I[ReLU激活]

--> J[最大池化 64@4x15]

--> K[全连接层 1024]

--> L[4个独立分类头]

--> M[4个字符预测]

训练结果

在1000张验证码上训练20个epoch后的结果：
Epoch Loss Validation Accuracy Best Accuracy

1 15.768 0.00% -
5 5.517 28.60% 28.60%
10 3.090 59.30% 59.30%
15 2.184 72.00% 72.00%
19 1.752 80.50% 80.50%
20 1.688 79.90% 80.50%

最终准确率：80.5%

自定义配置

修改字符集

在code_gen.py中修改characters变量：
默认字符集（数字+大写字母+小写字母）

characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

仅使用数字和大写字母

characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

调整训练参数

在vci_all.py或vci_train.py中修改：
训练轮数

for epoch in range(20):  # 修改为需要的epoch数

学习率

optimizer = optim.Adam(model.parameters(), lr=0.001)  # 调整学习率

批量大小

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

项目文件说明

主要脚本
code_gen.py  

   验证码生成脚本，用于创建训练数据集
支持自定义字符集

可调整生成数量

文件名格式：code_验证码_时间戳.png
vci_all.py  

   主脚本，包含：
数据加载与预处理

模型定义

训练与验证循环

预测功能
vci_model.py  

   模型架构定义（CNN分类器）
vci_train.py  

   训练脚本（精简版）
vci_predict.py  

   预测脚本，加载模型进行单图预测

数据目录
data/train：训练集验证码

data/test：测试集验证码（需手动创建）

贡献指南

欢迎贡献！请遵循以下步骤：
Fork 本仓库

创建您的分支 (git checkout -b feature/AmazingFeature)

提交您的更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

发起 Pull Request

许可证

本项目采用 LICENSE。

提示：在实际部署时，建议使用更大的数据集（1万张以上）进行训练以获得更好的识别效果。
