"""
YOLOv8 + CBAM 简易集成方法
无需修改ultralytics源码,通过自定义模块实现

适合毕业设计快速实现
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.nn.tasks import DetectionModel


# ============================================
# 步骤1: 定义CBAM模块
# ============================================

class CBAM(nn.Module):
    """CBAM注意力机制"""

    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        )

        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class C2f_CBAM(C2f):
    """
    继承C2f模块,添加CBAM
    这是最简单的集成方式
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)  # 在输出后添加CBAM

    def forward(self, x):
        # 调用父类的forward
        x = super().forward(x)
        # 应用CBAM
        x = self.cbam(x)
        return x


# ============================================
# 步骤2: 构建改进的YOLOv8模型
# ============================================

class YOLOv8_CBAM(nn.Module):
    """
    手动构建YOLOv8-CBAM模型
    """

    def __init__(self, nc=15, width_multiple=0.25, depth_multiple=0.33):
        super().__init__()

        # 计算通道数
        def make_divisible(x, divisor=8):
            return max(divisor, int(x + divisor / 2) // divisor * divisor)

        ch = [64, 128, 256, 512, 1024]
        ch = [make_divisible(c * width_multiple) for c in ch]

        # Backbone
        self.conv1 = Conv(3, ch[0], 3, 2)  # P1/2
        self.conv2 = Conv(ch[0], ch[1], 3, 2)  # P2/4
        self.c2f1 = C2f(ch[1], ch[1], n=round(3 * depth_multiple))  # 普通C2f

        self.conv3 = Conv(ch[1], ch[2], 3, 2)  # P3/8
        self.c2f2 = C2f_CBAM(ch[2], ch[2], n=round(6 * depth_multiple))  # ✨ 使用CBAM

        self.conv4 = Conv(ch[2], ch[3], 3, 2)  # P4/16
        self.c2f3 = C2f_CBAM(ch[3], ch[3], n=round(6 * depth_multiple))  # ✨ 使用CBAM

        self.conv5 = Conv(ch[3], ch[4], 3, 2)  # P5/32
        self.c2f4 = C2f_CBAM(ch[4], ch[4], n=round(3 * depth_multiple))  # ✨ 使用CBAM

        self.sppf = SPPF(ch[4], ch[4], k=5)

        print(f"✓ YOLOv8-CBAM模型构建完成")
        print(f"  - 通道配置: {ch}")
        print(f"  - CBAM层数: 3 (backbone)")

    def forward(self, x):
        # Backbone前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)

        p3 = self.c2f2(self.conv3(x))  # P3特征
        p4 = self.c2f3(self.conv4(p3))  # P4特征
        p5 = self.c2f4(self.conv5(p4))  # P5特征
        x = self.sppf(p5)

        return x


# ============================================
# 步骤3: 使用回调函数注入CBAM
# ============================================

def replace_c2f_with_cbam(model):
    """
    将模型中的C2f替换为C2f_CBAM
    这是最灵活的方法
    """

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, C2f):
            # 获取父模块和属性名
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model

            # 创建C2f_CBAM替换
            new_module = C2f_CBAM(
                module.cv1.conv.in_channels,
                module.cv2.conv.out_channels,
                n=len(module.m),
                shortcut=module.m[0].add if hasattr(module.m[0], 'add') else False
            )

            # 替换模块
            setattr(parent, attr_name, new_module)
            count += 1

    print(f"✓ 已替换 {count} 个C2f模块为C2f_CBAM")
    return model


# ============================================
# 步骤4: 训练脚本
# ============================================

def train_yolov8_cbam_method1():
    """
    方法1: 加载标准YOLOv8后替换模块
    推荐使用这种方法!
    """

    print("\n" + "="*60)
    print("方法1: 标准YOLOv8 + CBAM模块替换")
    print("="*60 + "\n")

    # 1. 加载标准YOLOv8
    model = YOLO('yolov8n.pt')

    # 2. 替换C2f为C2f_CBAM
    model.model = replace_c2f_with_cbam(model.model)

    # 3. 开始训练
    print("\n开始训练...")
    results = model.train(
        data='traffic_sign_data.yaml',
        epochs=100,
        imgsz=640,
        device='0',
        batch=16,
        name='yolov8n_cbam_method1',

        # 优化参数
        optimizer='AdamW',
        lr0=0.001,
        patience=20,

        # 数据增强
        augment=True,
        mosaic=1.0,
        mixup=0.1,

        cache=True,
        plots=True
    )

    print(f"\n✓ 训练完成!")
    return results


def train_yolov8_cbam_method2():
    """
    方法2: 从YAML配置加载(需要注册CBAM模块)
    """

    print("\n" + "="*60)
    print("方法2: 从YAML配置文件加载")
    print("="*60 + "\n")

    # 注册自定义模块
    from ultralytics.nn.tasks import attempt_load_one_weight

    # 这种方法需要修改ultralytics源码,不推荐
    print("⚠ 此方法需要修改源码,建议使用方法1")


def analyze_model_difference():
    """
    分析添加CBAM前后的模型差异
    """

    print("\n" + "="*60)
    print("模型结构对比分析")
    print("="*60 + "\n")

    # 基线模型
    model_baseline = YOLO('yolov8n.pt')

    # CBAM模型
    model_cbam = YOLO('yolov8n.pt')
    model_cbam.model = replace_c2f_with_cbam(model_cbam.model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    params_baseline = count_parameters(model_baseline.model)
    params_cbam = count_parameters(model_cbam.model)

    print(f"基线YOLOv8n参数量: {params_baseline:,}")
    print(f"YOLOv8n-CBAM参数量: {params_cbam:,}")
    print(f"参数增加: {params_cbam - params_baseline:,} ({(params_cbam/params_baseline - 1)*100:.2f}%)")

    # 推理速度测试
    import time
    dummy_input = torch.randn(1, 3, 640, 640)

    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model_baseline.model.cuda()
        model_cbam.model.cuda()

    # 预热
    for _ in range(10):
        _ = model_baseline.model(dummy_input)
        _ = model_cbam.model(dummy_input)

    # 测试基线模型
    times_baseline = []
    for _ in range(100):
        start = time.time()
        _ = model_baseline.model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_baseline.append(time.time() - start)

    # 测试CBAM模型
    times_cbam = []
    for _ in range(100):
        start = time.time()
        _ = model_cbam.model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_cbam.append(time.time() - start)

    avg_time_baseline = sum(times_baseline) / len(times_baseline) * 1000
    avg_time_cbam = sum(times_cbam) / len(times_cbam) * 1000

    print(f"\n推理速度对比 (640x640):")
    print(f"基线模型: {avg_time_baseline:.2f} ms ({1000/avg_time_baseline:.1f} FPS)")
    print(f"CBAM模型: {avg_time_cbam:.2f} ms ({1000/avg_time_cbam:.1f} FPS)")
    print(f"速度下降: {(avg_time_cbam/avg_time_baseline - 1)*100:.2f}%")


# ============================================
# 步骤5: 主程序
# ============================================

def main():
    """主函数"""

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'analyze'],
                        help='运行模式: train(训练) 或 analyze(分析)')
    args = parser.parse_args()

    if args.mode == 'train':
        # 训练CBAM模型
        train_yolov8_cbam_method1()

    elif args.mode == 'analyze':
        # 分析模型差异
        analyze_model_difference()


if __name__ == '__main__':
    # 直接运行训练
    train_yolov8_cbam_method1()

    # 或者分析模型
    # analyze_model_difference()