# 用标签控制扩散网络生成

import sys

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['DISPLAY'] = ":0"

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 1. 数据准备
# 加载MNIST数据集
dataset = torchvision.datasets.MNIST(
    root="./mnist/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
# 创建数据加载器（训练时用较大batch_size，这里保持原文档配置）
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 查看数据集样本（可选，用于验证数据加载）
def visualize_samples():
    x, y = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.figure(figsize=(8, 2))
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    plt.title('MNIST Sample Images')
    plt.axis('off')
    plt.show()

# 可视化样本（如果不需要可注释掉）
visualize_samples()

# 2. 定义类别条件UNet模型
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()
        # 类别嵌入层：将类别标签映射为特征向量
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        # 基础UNet模型（添加额外通道用于接收类别条件）
        self.model = UNet2DModel(
            sample_size=28,          # 生成图像尺寸（MNIST为28x28）
            in_channels=1 + class_emb_size,  # 输入通道：1（图像）+ 4（类别嵌入）
            out_channels=1,          # 输出通道（灰度图）
            layers_per_block=2,      # 每个block的残差层数
            block_out_channels=(32, 64, 64),  # 各block输出通道数
            down_block_types=(
                "DownBlock2D",        # 常规下采样块
                "AttnDownBlock2D",    # 带空间注意力的下采样块
                "AttnDownBlock2D",    # 带空间注意力的下采样块
            ),
            up_block_types=(
                "AttnUpBlock2D",      # 带空间注意力的上采样块
                "AttnUpBlock2D",      # 带空间注意力的上采样块
                "UpBlock2D",          # 常规上采样块
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        # 1. 将类别标签映射为嵌入向量
        class_emb = self.class_emb(class_labels)  # (bs, class_emb_size)
        print(class_emb.shape)
        # 2. 扩展维度以匹配图像尺寸（bs, 4, 28, 28）
        class_emb = class_emb.view(bs, class_emb.shape[1], 1, 1).expand(bs, class_emb.shape[1], w, h)
        print(class_emb.shape)
        # 3. 拼接图像和类别条件（输入通道变为1+4=5）
        net_input = torch.cat((x, class_emb), dim=1)
        # 4. 传入UNet模型预测噪声
        return self.model(net_input, t).sample  # 返回预测的噪声

# 3. 初始化模型、调度器、损失函数和优化器
# 噪声调度器（DDPM默认配置）
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# 初始化模型
net = ClassConditionedUnet().to(device)
# 损失函数（MSE损失，预测噪声与真实噪声的差距）
loss_fn = nn.MSELoss()
# 优化器（Adam优化器）
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# 4. 模型训练
def train_model(n_epochs=10):
    net.train()
    losses = []
    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')

        for x, y in progress_bar:
            # 数据预处理：移到设备上并归一化到[-1, 1]（扩散模型常用范围）
            x = x.to(device) * 2 - 1  # MNIST原范围[0,1] -> [-1,1]
            y = y.to(device)

            # 生成随机噪声（与输入图像同分布）
            noise = torch.randn_like(x)
            # 生成随机时间步（0~999）
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (x.shape[0],)).long().to(device)
            # 对图像添加噪声（前向扩散过程）
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # 模型预测噪声
            pred_noise = net(noisy_x, timesteps, y)
            # 计算损失
            loss = loss_fn(pred_noise, noise)

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"batch_loss":loss.item()})

            # 记录epoch平均损失
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.extend(epoch_losses)
            print(f'Finished epoch {epoch+1}. Average loss: {epoch_avg_loss:.5f}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    return losses

# 启动训练（默认10个epoch，可调整）
train_model(n_epochs=10)

# 5. 模型采样（生成指定类别的图像）
def generate_images(num_samples_per_class=8):
    net.eval()
    # 准备生成的类别：0~9每个类别生成num_samples_per_class张图
    classes = torch.tensor([i for i in range(10) for _ in range(num_samples_per_class)]).to(device)
    # 生成初始随机噪声（batch_size=10*8=80）
    x = torch.randn(len(classes), 1, 28, 28).to(device)

    print(f"Generating {num_samples_per_class} samples for each class (0-9)...")
    # 反向扩散过程：从噪声逐步去噪
    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc='Sampling'):
            # 预测残差（噪声）
            residual = net(x, t, classes)
            # 调度器更新：去噪一步
            x = noise_scheduler.step(residual, t, x).prev_sample

    # 数据后处理：从[-1,1]映射回[0,1]
    x = (x.detach().cpu() + 1) / 2
    # 裁剪到有效范围
    x = x.clip(0, 1)

    # 可视化生成结果
    plt.figure(figsize=(16, 10))
    grid = torchvision.utils.make_grid(x, nrow=num_samples_per_class, padding=2)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='Greys')
    plt.title('Generated MNIST Images (0-9, each class has 8 samples)')
    plt.axis('off')
    plt.show()

# 生成图像
generate_images(num_samples_per_class=8)