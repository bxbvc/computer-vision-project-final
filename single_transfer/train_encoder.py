import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
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


# ========================== 2. 重叠窗口工具 ==========================

def window_partition_overlap(x, window_size, stride):
    """
    重叠窗口划分。stride < window_size 时，相邻窗口共享像素。
    返回: (B, L, ws*ws, C), stride
    """
    B, C, H, W = x.shape
    ws = window_size
    x = F.unfold(x, kernel_size=ws, stride=stride)
    L = x.shape[-1]
    x = x.view(B, C, ws * ws, L)
    x = x.permute(0, 3, 2, 1).contiguous()
    return x, stride


def window_reverse_overlap(x, B, C, H, W, window_size, stride):
    """
    重叠窗口还原。F.fold 会把重叠区域的值累加，
    因此需要除以每个像素被覆盖的次数（norm）做平均。
    """
    ws = window_size
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(B, C * ws * ws, -1)
    out = F.fold(x, output_size=(H, W), kernel_size=ws, stride=stride)

    # 计算每个位置被多少个窗口覆盖
    ones = torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype)
    ones_unfold = F.unfold(ones, kernel_size=ws, stride=stride)
    norm = F.fold(ones_unfold, output_size=(H, W), kernel_size=ws, stride=stride)
    out = out / (norm + 1e-8)
    return out


# ========================== 3. 模型 ==========================
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

        # ---- 局部分支：Conv+BN+ReLU+Conv+BN 一条龙（替代 FFN）----
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        # ---- 窗口注意力分支 ----
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


# ========================== 4. 损失函数 ==========================

def compute_loss(img, z, tau=0.4, lambda_tv=0.1):
    """
    仅保留：
      1) L1：特征梯度匹配图像梯度
      2) TV：特征图 Total Variation 平滑（替代原有的 Grid Loss）
    """
    # ---- L1 ----
    img_r = img[:, :, :, 1:] - img[:, :, :, :-1]
    z_r   = z[:, :, :, 1:]   - z[:, :, :, :-1]
    d2_img_r = (img_r ** 2).sum(dim=1)
    d2_z_r   = (z_r   ** 2).sum(dim=1)
    target_r = torch.exp(d2_img_r - tau)
    loss1_r  = ((d2_z_r - target_r) ** 2).mean()

    img_d = img[:, :, 1:, :] - img[:, :, :-1, :]
    z_d   = z[:, :, 1:, :]   - z[:, :, :-1, :]
    d2_img_d = (img_d ** 2).sum(dim=1)
    d2_z_d   = (z_d   ** 2).sum(dim=1)
    target_d = torch.exp(d2_img_d - tau)
    loss1_d  = ((d2_z_d - target_d) ** 2).mean()

    loss1 = 0.5 * (loss1_r + loss1_d)

    # ---- TV Loss（特征平滑）----
    dx = z[:, :, :, 1:] - z[:, :, :, :-1]
    dy = z[:, :, 1:, :] - z[:, :, :-1, :]
    loss_tv = dx.abs().mean() + dy.abs().mean()

    loss = loss1 + lambda_tv * loss_tv
    return loss, {"L1": loss1.item(), "TV": loss_tv.item()}


# ========================== 5. 可视化 ==========================

@torch.no_grad()
def pca_project(z):
    C, H, W = z.shape
    x = z.reshape(C, -1).T
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    U, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)
    proj = x_centered @ Vt[:3].T
    proj = proj - proj.min(dim=0, keepdim=True)[0]
    proj = proj / (proj.max(dim=0, keepdim=True)[0] + 1e-8)
    return proj.T.reshape(3, H, W)


@torch.no_grad()
def kmeans_cluster(z, K=8, max_iter=10, merge_thresh=0.8):
    C, H, W = z.shape
    x = z.reshape(C, -1).T
    N = x.shape[0]
    idx = torch.randperm(N, device=x.device)[:K]
    centers = x[idx].clone()

    for _ in range(max_iter):
        dists = torch.cdist(x, centers)
        labels = dists.argmin(dim=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = x[mask].mean(dim=0)

    labels = labels.reshape(H, W)

    valid_centers = []
    valid_ids = []
    for k in range(K):
        mask = labels == k
        if mask.any():
            mean_feat = z[:, mask].mean(dim=1)
            valid_centers.append(mean_feat)
            valid_ids.append(k)

    M = len(valid_ids)
    if M < 2:
        return labels

    valid_centers = torch.stack(valid_centers)
    valid_centers_norm = F.normalize(valid_centers, p=2, dim=1)
    sim_matrix = valid_centers_norm @ valid_centers_norm.T

    parent = list(range(M))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(M):
        for j in range(i + 1, M):
            if sim_matrix[i, j] > merge_thresh:
                union(i, j)

    root_to_new = {}
    new_label = 0
    old_to_new = {}
    for i in range(M):
        root = find(i)
        if root not in root_to_new:
            root_to_new[root] = new_label
            new_label += 1
        old_to_new[valid_ids[i]] = root_to_new[root]

    remapped = torch.zeros_like(labels)
    for old_id, new_id in old_to_new.items():
        remapped[labels == old_id] = new_id

    return remapped


def label_color(labels):
    n = labels.max().item() + 1
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, (max(n, 1), 3), dtype=np.uint8)
    return colors[labels.cpu().numpy()]


@torch.no_grad()
def save_vis(img_tensor, z_tensor, path="vis.png"):
    orig = img_tensor.cpu().numpy().transpose(1, 2, 0)
    orig = np.clip((orig + 1.0) / 2.0, 0, 1)
    orig_uint8 = (orig * 255).astype(np.uint8)

    pca = pca_project(z_tensor).cpu().numpy().transpose(1, 2, 0)
    pca_uint8 = np.clip(pca * 255, 0, 255).astype(np.uint8)

    labels_z = kmeans_cluster(z_tensor, K=12, max_iter=10, merge_thresh=0.95)
    seg_z = label_color(labels_z)

    labels_rgb = kmeans_cluster(img_tensor, K=12, max_iter=10, merge_thresh=1)
    seg_rgb = label_color(labels_rgb)

    H, W = orig_uint8.shape[:2]
    canvas = Image.new("RGB", (W * 4, H))
    canvas.paste(Image.fromarray(orig_uint8),  (0,     0))
    canvas.paste(Image.fromarray(pca_uint8),   (W,     0))
    canvas.paste(Image.fromarray(seg_z),       (W * 2, 0))
    canvas.paste(Image.fromarray(seg_rgb),     (W * 3, 0))
    canvas.save(path)


# ========================== 6. 训练 ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    coco_root = r"data\coco\train2017"

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = COCORaw(coco_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    win_size = 4
    model = TextureSegmentor(dim=18, window_size=win_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_latest = ckpt_dir / "encoder.pth"
    start_epoch = 1
    if ckpt_latest.exists():
        ckp = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckp["model_state_dict"])
        optimizer.load_state_dict(ckp["optimizer_state_dict"])
        start_epoch = ckp.get("epoch", 0) + 1
        print(f"Resume epoch {start_epoch}")
    else:
        print("Start from scratch")

    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_dict = {"L1": 0.0, "TV": 0.0}

    for epoch in range(start_epoch, 201):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for imgs in pbar:
            imgs = imgs.to(device)
            z = model(imgs)

            loss, loss_dict = compute_loss(
                imgs, z,
                tau=0.3,
                lambda_tv=1,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            accum_loss += loss.item()
            for k in accum_dict:
                accum_dict[k] += loss_dict[k]

            if global_step % 15 == 0:
                avg_loss = accum_loss / 15
                avg_L1 = accum_dict["L1"] / 15
                avg_TV = accum_dict["TV"] / 15
                print(f"\n[Step {global_step}] Avg Loss: {avg_loss:.4f} | L1: {avg_L1:.4f} | TV: {avg_TV:.4f}")

                save_vis(imgs[0], z[0], path=f"vis.png")

                accum_loss = 0.0
                for k in accum_dict:
                    accum_dict[k] = 0.0

            if global_step % 100 == 0:
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch / global_step,
                }
                torch.save(state, ckpt_latest)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "L1":   f"{loss_dict['L1']:.4f}",
                "TV":   f"{loss_dict['TV']:.4f}",
            })

        save_vis(imgs[0], z[0], path="vis_latest.png")

    print("Done.")


if __name__ == "__main__":
    main()