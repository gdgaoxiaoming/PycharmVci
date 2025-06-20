import os
import random
import time

from PIL import Image, ImageDraw, ImageFont

# 创建输出目录
os.makedirs('./data/train', exist_ok=True)

# 定义字符集（数字0-9 + 大写A-Z）
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# 生成1000张验证码
for i in range(10):  # 修正为1000张
    # 生成4位随机验证码
    captcha_text = ''.join(random.choices(characters, k=4))

    # 生成当前时间戳
    timestamp = int(time.time() * 1000)

    # 生成文件名
    filename = f'code_{captcha_text}_{timestamp}.png'
    filepath = os.path.join('./data/train', filename)

    # 创建简单验证码图像
    image = Image.new('RGB', (120, 32), color=(255, 255, 255))  # 白色背景
    draw = ImageDraw.Draw(image)

    # 使用清晰字体（确保系统有Arial字体，或替换为其他字体路径）
    #font = ImageFont.truetype("arial.ttf", 24)  # 增大字体大小
    try:
        # 尝试加载 Arial（仅当系统已安装时生效）
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        # 教学时优先保证运行成功
        font = ImageFont.load_default()
        #print("警告：使用默认字体，建议安装 Arial 获得更好效果")

    # ✅ 修复点：使用 textbbox 替代 textsize
    left, top, right, bottom = draw.textbbox((0, 0), captcha_text, font=font)
    text_width, text_height = right - left, bottom - top

    # 绘制验证码文本（居中显示）
    position = ((120 - text_width) / 2, (32 - text_height) / 2 - 2)  # 垂直居中
    draw.text(position, captcha_text, font=font, fill=(0, 0, 0))  # 黑色文本

    # 保存图片
    image.save(filepath)

    # 每生成100张显示进度
    if (i + 1) % 100 == 0:
        print(f'已生成 {i + 1}/1000 张验证码')

print('验证码生成完成！所有图片已保存至 ./data/train 目录')