import os
import random
from pathlib import Path
import time
import csv
from glob import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SEED = 42
NUM_PER_STYLE = 30
BINS = 33
N_ITER = 800
LR = 0.001
TV_WEIGHT = 10

CACHE_DIR = Path('cache_gram')
COCO_CACHE = CACHE_DIR / 'coco_files.pkl'
WIKI_CACHE = CACHE_DIR / 'wiki_classes.pkl'


def fmt4(x):
    return f"{x:.4g}"


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


def transfer_color_lut(content, prob, style_img, bins=33, n_iter=800, lr=0.001, tv_weight=10):
    device = content.device
    B, C, H, W = content.shape
    assert B == 1

    with torch.no_grad():
        style_01 = (style_img + 1.0) / 2.0
        P_style = splat_3d(style_01, bins).squeeze(0)
        P_target = 0.5 * prob + 0.5 * P_style

    t = torch.linspace(0, 1, bins, device=device)
    r = t.view(bins, 1, 1, 1).expand(bins, bins, bins, 1)
    g = t.view(1, bins, 1, 1).expand(bins, bins, bins, 1)
    b = t.view(1, 1, bins, 1).expand(bins, bins, bins, 1)
    identity = torch.cat([r, g, b], dim=-1)

    delta = torch.zeros_like(identity, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    content_01 = (content + 1.0) / 2.0
    grid = content_01.permute(0, 2, 3, 1).unsqueeze(1) * 2.0 - 1.0

    for it in range(n_iter):
        optimizer.zero_grad()
        lut = (identity + delta).clamp(0, 1)
        lut_gs = lut.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        out = out.squeeze(2)
        H_dist = splat_3d(out, bins).squeeze(0)
        H_safe = H_dist + 1e-8
        P_safe = P_target + 1e-8
        kl = (H_safe * (H_safe.log() - P_safe.log())).sum()

        tv_r = torch.abs(lut[1:,:,:,:] - lut[:-1,:,:,:]).mean()
        tv_g = torch.abs(lut[:,1:,:,:] - lut[:,:-1,:,:]).mean()
        tv_b = torch.abs(lut[:,:,1:,:] - lut[:,:,:-1,:]).mean()
        tv_lut = tv_r + tv_g + tv_b

        loss = kl + tv_weight * tv_lut
        loss.backward()
        optimizer.step()

        if it % 400 == 0 or it == n_iter - 1:
            print(f"  [Color LUT] Iter {it+1}/{n_iter}: KL = {kl.item():.4f}")

    with torch.no_grad():
        lut_final = (identity + delta).clamp(0, 1)
        lut_gs = lut_final.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear',
                            padding_mode='border', align_corners=True).squeeze(2)
        return out * 2.0 - 1.0


class VGGStyleLoss(nn.Module):
    def __init__(self, vgg_state_dict_path):
        super().__init__()
        vgg = models.vgg19()
        vgg.load_state_dict(torch.load(vgg_state_dict_path, map_location="cpu"))
        vgg.eval()
        self.features = vgg.features
        for p in self.features.parameters():
            p.requires_grad = False

        self.style_layers = [1, 6, 11, 20]

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _gram_matrix(self, feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G / (c * h * w)

    def forward(self, generated, style):
        gen = self._normalize(generated)
        sty = self._normalize(style)

        x = gen
        gen_feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                gen_feats[i] = x

        style_loss = 0
        x = sty
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                style_loss += F.mse_loss(self._gram_matrix(gen_feats[i]), self._gram_matrix(x))

        return style_loss


def fast_list_images(root, cache_file=None, exts=None):
    import pickle
    if exts is None:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    root = str(root)
    files = []
    if os.path.isdir(root):
        try:
            files = [os.path.join(root, f) for f in os.listdir(root)
                     if os.path.splitext(f)[1].lower() in exts]
        except PermissionError:
            files = []
        if not files:
            patterns = [os.path.join(root, '**', f'*{ext}') for ext in exts]
            for pat in patterns:
                files.extend(glob(pat, recursive=True))
    files = sorted(set(files))
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    return files


def get_wikiart_by_classes(wiki_root, cache_file=None):
    import pickle
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    wiki_root = Path(wiki_root)
    classes = {}
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    for split in ['train', 'test']:
        sp = wiki_root / split
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            class_name = d.name
            files = [str(f) for f in d.iterdir() if f.suffix.lower() in exts]
            if files:
                if class_name not in classes:
                    classes[class_name] = []
                classes[class_name].extend(files)

    for k in classes:
        classes[k] = sorted(set(classes[k]))

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(classes, f)
    return classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}, Seed: {SEED}, Num per style: {NUM_PER_STYLE}")

    out_dir = Path("color_benchmark")
    out_dir.mkdir(exist_ok=True)

    coco_root = r"data\coco\train2017"
    wiki_root = r"data\wikiart_images"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ckpt_color = Path("checkpoints/latest.pth")
    if not ckpt_color.exists():
        raise FileNotFoundError(f"Color checkpoint not found: {ckpt_color}")
    color_model = StyleColorPredictor(bins=BINS).to(device)
    color_model.load_state_dict(torch.load(ckpt_color, map_location=device)["model"])
    color_model.eval()
    for p in color_model.parameters():
        p.requires_grad = False

    vgg_path = Path("checkpoints/vgg.pth")
    if not vgg_path.exists():
        print("Downloading VGG19...")
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(vgg.state_dict(), vgg_path)
    vgg_style = VGGStyleLoss(str(vgg_path)).to(device)

    coco_imgs = fast_list_images(coco_root, cache_file=str(COCO_CACHE))
    if len(coco_imgs) == 0:
        raise RuntimeError("No COCO images found")
    random.shuffle(coco_imgs)

    wiki_classes = get_wikiart_by_classes(wiki_root, cache_file=str(WIKI_CACHE))
    if not wiki_classes:
        raise RuntimeError("No WikiArt classes found")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["style_name", "total_sec", "direct_loss", "stylized_loss"])

    coco_ptr = 0
    for class_name, style_files in sorted(wiki_classes.items()):
        if not style_files:
            continue

        if coco_ptr + NUM_PER_STYLE > len(coco_imgs):
            random.shuffle(coco_imgs)
            coco_ptr = 0
        content_batch = coco_imgs[coco_ptr:coco_ptr + NUM_PER_STYLE]
        coco_ptr += NUM_PER_STYLE

        print(f"\n{'='*60}")
        print(f"Style: {class_name} | Processing {NUM_PER_STYLE} content images")
        print(f"{'='*60}")

        for i in range(NUM_PER_STYLE):
            content_path = content_batch[i]
            style_path = style_files[i % len(style_files)]

            content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
            style_img = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                direct_loss = vgg_style(content_img, style_img).item()

            t_start = time.perf_counter()

            with torch.no_grad():
                prob, _ = color_model(style_img)

            I_color = transfer_color_lut(content_img, prob[0], style_img,
                                         bins=BINS, n_iter=N_ITER, lr=LR, tv_weight=TV_WEIGHT)

            t_end = time.perf_counter()
            total_sec = t_end - t_start

            with torch.no_grad():
                stylized_loss = vgg_style(I_color, style_img).item()

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    class_name,
                    fmt4(total_sec),
                    fmt4(direct_loss),
                    fmt4(stylized_loss)
                ])

            print(f"  [{i+1:03d}/{NUM_PER_STYLE}] Total={fmt4(total_sec)}s | "
                  f"Direct={fmt4(direct_loss)} | Stylized={fmt4(stylized_loss)}")

    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()