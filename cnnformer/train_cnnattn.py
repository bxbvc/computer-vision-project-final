import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ========================== 注意力模型定义 ==========================

class AttentionLayer(nn.Module):
    """
    单层“奇怪”注意力（主干网络 + 1 + 2）：
      主干：3×3 卷积生成 Q/V，下采样残差；
      1  ：4×4 常规窗口局部注意力；
      2  ：4×4 偏移一半（shift=2）窗口局部注意力；
      最后：残差 x + out1 + out2 → BN∘GELU。
    """
    def __init__(self, in_channels: int, transform: bool):
        super().__init__()
        self.transform = transform

        if transform:
            out_channels = in_channels * 2
            stride = 2
        else:
            out_channels = in_channels
            stride = 1

        self.gelu = nn.GELU()

        # ---------- 主干网络 ----------
        self.q_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.q_bn1 = nn.BatchNorm2d(out_channels)
        self.q_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.q_bn2 = nn.BatchNorm2d(out_channels)

        self.v_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.v_bn1 = nn.BatchNorm2d(out_channels)
        self.v_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.v_bn2 = nn.BatchNorm2d(out_channels)

        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_down = nn.BatchNorm2d(out_channels)

    def _window_attn(self, Q, V, h, w, token_len, shift):
        """4×4 窗口局部注意力，支持循环偏移（shift）"""
        B = Q.size(0)
        num_win_h = h // 4
        num_win_w = w // 4

        # 循环偏移（偏移一半窗口）
        if shift > 0:
            Q = torch.roll(Q, shifts=(shift, shift), dims=(2, 3))
            V = torch.roll(V, shifts=(shift, shift), dims=(2, 3))

        # (B, token_len, h, w) -> (B, num_windows, 16, token_len)
        def to_windows(t):
            t = t.view(B, token_len, num_win_h, 4, num_win_w, 4)
            t = t.permute(0, 2, 4, 3, 5, 1)
            t = t.reshape(B, num_win_h * num_win_w, 16, token_len)
            return t

        Q_win = to_windows(Q)
        V_win = to_windows(V)

        attn = torch.matmul(Q_win, Q_win.transpose(-2, -1)) / math.sqrt(token_len)
        A = F.softmax(attn, dim=-1)
        out = torch.matmul(A, V_win)

        # 恢复空间形状
        out = out.view(B, num_win_h, num_win_w, 4, 4, token_len)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, token_len, h, w)

        # 逆循环偏移
        if shift > 0:
            out = torch.roll(out, shifts=(-shift, -shift), dims=(2, 3))

        return out

    def forward(self, x):
        B, C, H, W = x.shape

        if self.transform:
            token_len = C * 2
            h = H // 2
            w = W // 2
        else:
            token_len = C
            h = H
            w = W

        assert h % 4 == 0 and w % 4 == 0, f"特征图尺寸 ({h},{w}) 必须能被 4 整除"

        # ==================== 主干网络 ====================
        Q = self.gelu(self.q_bn1(self.q_conv1(x)))
        Q = self.q_bn2(self.q_conv2(Q))

        V = self.gelu(self.v_bn1(self.v_conv1(x)))
        V = self.v_bn2(self.v_conv2(V))

        x = self.down_conv(x)

        # ==================== 1: 常规窗口注意力 ====================
        out1 = self._window_attn(Q, V, h, w, token_len, shift=0)

        # ==================== 2: 偏移一半窗口注意力 ====================
        out2 = self._window_attn(Q, V, h, w, token_len, shift=2)

        # ==================== 融合 ====================
        out = x + out1 + out2
        out = self.gelu(self.bn_down(out))
        return out


class FinalAttentionLayer(nn.Module):
    def __init__(self, in_channels, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


class AttentionResNet(nn.Module):
    """
    整体架构（通道逐层翻倍）：
      conv1(3→64, 3×3)      → 64×32×32
      layer1 (2×2 Conv)       → 128×16×16
      layer2 (2×2 Conv)       → 256×8×8
      layer3 (2×2 Conv)       → 512×4×4
      final(QᵀK 注意力)       → 100
    """
    def __init__(self, channels=64, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.layer1 = AttentionLayer(channels, True)
        self.layer2 = AttentionLayer(channels * 2, False)
        self.layer3 = AttentionLayer(channels * 2, True)
        self.layer4 = AttentionLayer(channels * 4, False)
        self.layer5 = AttentionLayer(channels * 4, True)

        self.final = FinalAttentionLayer(channels * 8, num_classes)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        logits = self.final(x)
        return logits


# ========================== 训练工具 ==========================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100. * correct / total


# ========================== 主函数 ==========================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CIFAR-100 标准预处理
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    model = AttentionResNet(channels=48, num_classes=100).to(device)
    print(f"Model: AttentionResNet | Params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs('checkpoint', exist_ok=True)

    for epoch in range(1, 201):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(f"Epoch [{epoch:03d}/200] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.4f}")

        # 每个 epoch 保存当前独一无二的 ckpt 到 checkpoint/ct.pth，仅保存模型参数
        torch.save(model.state_dict(), 'checkpoint/ct.pth')

        if epoch % 10 == 0 or epoch == 200:
            val_loss, val_acc = evaluate(model, testloader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            print(f"{'-'*50}")
            print(f"Epoch [{epoch:03d}/200] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
            print(f"{'-'*50}")

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"Training Complete | Best Val Acc: {best_val_acc:.2f}% | Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()