import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ========================== 1. 数据集 ==========================

class COCORaw(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.imgs = [p for p in self.root.iterdir() if p.suffix.lower() in exts]
        self.imgs.sort()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ========================== 2. 重叠窗口工具 (与 Encoder 完全一致) ==========================

def window_partition_overlap(x, window_size, stride):
    B, C, H, W = x.shape
    ws = window_size
    x = F.unfold(x, kernel_size=ws, stride=stride)
    L = x.shape[-1]
    x = x.view(B, C, ws * ws, L)
    x = x.permute(0, 3, 2, 1).contiguous()
    return x, stride


def window_reverse_overlap(x, B, C, H, W, window_size, stride):
    ws = window_size
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(B, C * ws * ws, -1)
    out = F.fold(x, output_size=(H, W), kernel_size=ws, stride=stride)

    ones = torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype)
    ones_unfold = F.unfold(ones, kernel_size=ws, stride=stride)
    norm = F.fold(ones_unfold, output_size=(H, W), kernel_size=ws, stride=stride)
    out = out / (norm + 1e-8)
    return out


# ========================== 3. Encoder (最新 3-block 版本，带 forward_with_skips) ==========================

class MultiScaleStem(nn.Module):
    """
    RGB 多尺度最大池化 Stem（无参）。
    对 RGB 3 个通道分别做步长 1,2,4,8,16,32 的最大池化，
    再双线性插值回原尺寸，拼接为 18 通道。
    """
    def __init__(self, out_dim=18):
        super().__init__()
        assert out_dim == 18, "out_dim must be 18 (3 RGB channels x 6 scales)"
        self.scales = [1, 2, 4, 8, 16, 32]

    def forward(self, x):
        # x 经过 Normalize(mean=0.5, std=0.5)，范围 [-1, 1]；先还原到 [0, 1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        B, C, H, W = x.shape
        feats = []
        for s in self.scales:
            if s == 1:
                pooled = x
            else:
                pooled = F.max_pool2d(x, kernel_size=s, stride=s)
                pooled = F.interpolate(
                    pooled, size=(H, W), mode='bilinear', align_corners=False
                )
            feats.append(pooled)

        out = torch.cat(feats, dim=1)  # (B, 18, H, W)
        return out


class CNNTransformerBlock(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.dim = dim
        self.win = window_size
        self.shift = window_size // 2
        self.stride = window_size // 2  # 50% 重叠，彻底消除 fold 接缝

        # 局部分支：Conv+BN+ReLU+Conv+BN 一条龙（替代 FFN）
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # 窗口注意力分支
        self.q_conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn1 = nn.BatchNorm2d(dim)
        self.q_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn2 = nn.BatchNorm2d(dim)

        self.out_norm = nn.BatchNorm2d(dim)

    def _window_attn(self, Q):
        B, C, H, W = Q.shape
        Q_w, stride = window_partition_overlap(Q, self.win, self.stride)
        attn = torch.matmul(Q_w, Q_w.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Q_w)
        out = window_reverse_overlap(out, B, C, H, W, self.win, stride)
        return out

    def forward(self, x):
        # 局部分支（局部卷积，纯 CNN 行为）
        local = self.local(x)

        # 注意力分支（含 shifted window）
        Q = self.q_bn2(self.q_conv2(F.relu(self.q_bn1(self.q_conv1(x)))))
        B1 = self._window_attn(Q)

        Q_shifted = torch.roll(Q, shifts=(self.shift, self.shift), dims=(2, 3))
        B2 = self._window_attn(Q_shifted)
        B2 = torch.roll(B2, shifts=(-self.shift, -self.shift), dims=(2, 3))

        attn = B1 + B2

        # 残差融合：原始 + 局部 + 注意力
        out = F.relu(self.out_norm(x + local + attn))
        return out


class TextureSegmentor(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.stem = MultiScaleStem(out_dim=dim)
        self.block1 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block2 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block3 = CNNTransformerBlock(dim=dim, window_size=window_size)

    def forward(self, x):
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return z

    @torch.no_grad()
    def forward_with_skips(self, x):
        """为 Decoder 提供跳跃连接：skips=[stem, block1, block2]"""
        e1 = self.stem(x)
        e2 = self.block1(e1)
        e3 = self.block2(e2)
        z = self.block3(e3)
        return z, [e1, e2, e3]


# ========================== 4. Decoder (适配 3-block Encoder) ==========================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3):
        super().__init__()
        # 为每个跳跃连接创建解码块，从深到浅依次融合
        self.blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        for _ in range(num_skips):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, hid_ch))
            in_ch = hid_ch

        # 最后的精修卷积
        self.refine = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1, bias=True),
            nn.Tanh(),  # 输出 [-1, 1]，与 Normalize(mean=0.5, std=0.5) 对应
        )

    def forward(self, z, skips):
        # skips = [stem, block1, block2] (从浅到深)
        # 解码时从深到浅融合
        x = z
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)
        x = self.refine(x)
        return self.head(x)


# ========================== 5. 可视化 ==========================

@torch.no_grad()
def save_recon_vis(img_tensor, rec_tensor, path="recon_vis.png"):
    def to_uint8(t):
        arr = t.cpu().numpy().transpose(1, 2, 0)
        arr = np.clip((arr + 1.0) / 2.0, 0, 1)
        return (arr * 255).astype(np.uint8)

    orig = to_uint8(img_tensor)
    recon = to_uint8(rec_tensor)
    H, W = orig.shape[:2]
    canvas = Image.new("RGB", (W * 2, H))
    canvas.paste(Image.fromarray(orig), (0, 0))
    canvas.paste(Image.fromarray(recon), (W, 0))
    canvas.save(path)


# ========================== 6. 训练 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coco_root = r"data\coco\train2017"
    encoder_ckpt = Path("checkpoints") / "encoder.pth"
    decoder_ckpt_dir = Path("checkpoints")
    decoder_ckpt_dir.mkdir(exist_ok=True)
    decoder_ckpt = decoder_ckpt_dir / "decoder.pth"

    batch_size = 24
    epochs = 200
    lr = 1e-3

    # 与 Encoder 训练时保持一致 (192, 192)
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = COCORaw(coco_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    win_size = 4  # 必须与 Encoder 训练时一致
    encoder = TextureSegmentor(dim=18, window_size=win_size).to(device)
    decoder = Decoder(bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3).to(device)

    # 加载 Encoder 预训练权重（冻结）
    if encoder_ckpt.exists():
        ckp = torch.load(encoder_ckpt, map_location=device)
        encoder.load_state_dict(ckp["model_state_dict"])
        print(f"Loaded encoder from epoch {ckp.get('epoch', 'unknown')}")
    else:
        print("Warning: Encoder checkpoint not found, using random init")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {n_enc:,} (frozen)")
    print(f"Decoder params: {n_dec:,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.MSELoss()

    start_epoch = 1
    if decoder_ckpt.exists():
        try:
            ckp = torch.load(decoder_ckpt, map_location=device)
            decoder.load_state_dict(ckp["model_state_dict"])
            optimizer.load_state_dict(ckp["optimizer_state_dict"])
            start_epoch = ckp.get("epoch", 0) + 1
            print(f"Resume decoder from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Incompatible old checkpoint: {e}")
            print("Training new decoder from scratch.")

    decoder.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        epoch_loss = 0.0
        accum_loss = 0.0

        for imgs in pbar:
            imgs = imgs.to(device)

            with torch.no_grad():
                z, skips = encoder.forward_with_skips(imgs)

            rec = decoder(z, skips)
            loss = criterion(rec, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            accum_loss += loss.item()
            pbar.set_postfix({"mse": f"{loss.item():.6f}"})

            if global_step % 30 == 0:
                avg_mse = accum_loss / 30
                print(f"\n[Step {global_step}] Avg MSE: {avg_mse:.6f}")
                save_recon_vis(imgs[0], rec[0], path="recon_vis.png")
                accum_loss = 0.0

            if global_step % 100 == 0:
                state = {
                    "epoch": epoch,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch / global_step,
                }
                torch.save(state, decoder_ckpt)

        avg_loss = epoch_loss / len(loader)
        scheduler.step()

        with torch.no_grad():
            z_vis, skips_vis = encoder.forward_with_skips(imgs[:1])
            rec_vis = decoder(z_vis, skips_vis)
            save_recon_vis(imgs[0], rec_vis[0], path="recon_vis.png")

        if epoch % 10 == 0:
            torch.save(state, decoder_ckpt_dir / f"epoch_{epoch:03d}.pth")

        print(f"Epoch {epoch:03d} | Avg MSE: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, decoder_ckpt_dir / "best.pth")

    print(f"\nDone. Best MSE: {best_loss:.6f}")


if __name__ == "__main__":
    main()