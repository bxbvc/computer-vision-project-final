import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


# ========================== 1. 颜色迁移模型 ==========================

class StyleColorPredictor(nn.Module):
    def __init__(self, bins=33):
        super().__init__()
        self.bins = bins
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, bins ** 3)
        )

    def forward(self, style_img):
        x = (style_img + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.features(x).view(x.size(0), -1)
        logits = self.fc(feat)
        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp().view(-1, self.bins, self.bins, self.bins)
        return prob, log_prob


# ========================== 2. 3D 颜色分布工具 ==========================

SPLAT_OFFSETS = torch.tensor([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    [0,0,1], [1,0,1], [0,1,1], [1,1,1]
], dtype=torch.long)


def splat_3d(x, bins):
    x = x.clamp(0, 1)
    B, C, H, W = x.shape
    N = H * W
    x = x.permute(0, 2, 3, 1).reshape(B, N, 3)
    pos = x * (bins - 1)
    pos = pos.clamp(0, bins - 1 - 1e-4)
    base = pos.floor().long()
    frac = pos - base.float()

    out = torch.zeros(B, bins * bins * bins, device=x.device, dtype=x.dtype)
    offsets = SPLAT_OFFSETS.to(x.device).view(1, 1, 8, 3)
    idx_all = base.unsqueeze(2) + offsets
    idx_all = idx_all.clamp(0, bins - 1)
    flat_all = idx_all[..., 0] * bins * bins + idx_all[..., 1] * bins + idx_all[..., 2]

    w_all = torch.stack([
        (1 - frac[..., 0]) * (1 - frac[..., 1]) * (1 - frac[..., 2]),
        frac[..., 0]       * (1 - frac[..., 1]) * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * frac[..., 1]       * (1 - frac[..., 2]),
        frac[..., 0]       * frac[..., 1]       * (1 - frac[..., 2]),
        (1 - frac[..., 0]) * (1 - frac[..., 1]) * frac[..., 2],
        frac[..., 0]       * (1 - frac[..., 1]) * frac[..., 2],
        (1 - frac[..., 0]) * frac[..., 1]       * frac[..., 2],
        frac[..., 0]       * frac[..., 1]       * frac[..., 2],
    ], dim=2)

    out = torch.scatter_add(out, 1, flat_all.reshape(B, N * 8), w_all.reshape(B, N * 8))
    return out.view(B, bins, bins, bins) / N


def compute_kl(content_01, prob, bins=33):
    H = splat_3d(content_01, bins).squeeze(0)
    H = H + 1e-8
    P = prob + 1e-8
    return (H * (H.log() - P.log())).sum().item()


# ========================== 3. 主流程：逐类统计 KL（均值+方差，100张）==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wikiart_root = r"data\wikiart_images"
    ckpt_color   = Path("checkpoints/latest.pth")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---- 加载模型 ----
    color_model = StyleColorPredictor(bins=33).to(device)
    if not ckpt_color.exists():
        raise FileNotFoundError(f"Color checkpoint not found: {ckpt_color}")
    color_model.load_state_dict(torch.load(ckpt_color, map_location=device)["model"])
    color_model.eval()

    # ---- 遍历 WikiArt 类别 ----
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    results = []

    for split in ["train", "test"]:
        split_path = Path(wikiart_root) / split
        if not split_path.exists():
            continue

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
            imgs.sort()

            if len(imgs) == 0:
                continue

            # 取 100 张，不足则全取
            selected = imgs[:100]
            n = len(selected)

            kl_list = []
            for p in selected:
                img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                img_01 = (img + 1.0) / 2.0

                with torch.no_grad():
                    prob, _ = color_model(img)
                    kl = compute_kl(img_01, prob[0], bins=33)

                kl_list.append(kl)

            kl_array = np.array(kl_list)
            avg_kl = float(kl_array.mean())
            var_kl = float(kl_array.var(ddof=0))

            results.append((class_name, n, avg_kl, var_kl))
            print(f"Class: {class_name:20s} | Count: {n:3d} | Avg KL: {avg_kl:.6f} | Var KL: {var_kl:.6f}")

    print("\n========== Summary ==========")
    print(f"{'Class':<<20s} {'N':>4s} {'Avg KL':>12s} {'Var KL':>12s}")
    print("-" * 52)
    for name, n, avg_kl, var_kl in results:
        print(f"{name:<20s} {n:4d} {avg_kl:12.6f} {var_kl:12.6f}")


if __name__ == "__main__":
    main()