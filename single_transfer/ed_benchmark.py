# ed_benchmark.py
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

SEED = 42
NUM_PER_STYLE = 30
BATCH_SIZE = 4
ALPHA_Q = 1.0
POOL_SIZE = 64

CACHE_DIR = Path('cache_gram')
COCO_CACHE = CACHE_DIR / 'coco_files.pkl'
WIKI_CACHE = CACHE_DIR / 'wiki_classes.pkl'


def fmt4(x):
    return f"{x:.4g}"


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


class MultiScaleStem(nn.Module):
    def __init__(self, out_dim=18):
        super().__init__()
        assert out_dim == 18, "out_dim must be 18"
        self.scales = [1, 2, 4, 8, 16, 32]

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        B, C, H, W = x.shape
        feats = []
        for s in self.scales:
            if s == 1:
                pooled = x
            else:
                pooled = F.max_pool2d(x, kernel_size=s, stride=s)
                pooled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(pooled)
        return torch.cat(feats, dim=1)


class CNNTransformerBlock(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.dim = dim
        self.win = window_size
        self.shift = window_size // 2
        self.stride = window_size // 2

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.q_conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn1 = nn.BatchNorm2d(dim)
        self.q_conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.q_bn2 = nn.BatchNorm2d(dim)
        self.out_norm = nn.BatchNorm2d(dim)

    def get_q(self, x):
        return self.q_bn2(self.q_conv2(F.relu(self.q_bn1(self.q_conv1(x)))))

    def _window_attn(self, Q):
        B, C, H, W = Q.shape
        Q_w, stride = window_partition_overlap(Q, self.win, self.stride)
        attn = torch.matmul(Q_w, Q_w.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Q_w)
        out = window_reverse_overlap(out, B, C, H, W, self.win, stride)
        return out

    def forward(self, x):
        local = self.local(x)
        Q = self.get_q(x)
        B1 = self._window_attn(Q)
        Q_shifted = torch.roll(Q, shifts=(self.shift, self.shift), dims=(2, 3))
        B2 = self._window_attn(Q_shifted)
        B2 = torch.roll(B2, shifts=(-self.shift, -self.shift), dims=(2, 3))
        attn = B1 + B2
        out = F.relu(self.out_norm(x + local + attn))
        return out


class TextureSegmentor(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.stem = MultiScaleStem(out_dim=dim)
        self.block1 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block2 = CNNTransformerBlock(dim=dim, window_size=window_size)
        self.block3 = CNNTransformerBlock(dim=dim, window_size=window_size)

    @torch.no_grad()
    def forward_with_skips(self, x):
        e1 = self.stem(x)
        e2 = self.block1(e1)
        e3 = self.block2(e2)
        z = self.block3(e3)
        return z, [e1, e2, e3]

    @torch.no_grad()
    def forward_with_skips_and_style(self, x, q_styles):
        e1 = self.stem(x)
        e2 = self.block1.forward_with_style(e1, q_styles[0]) if q_styles[0] is not None else self.block1(e1)
        e3 = self.block2.forward_with_style(e2, q_styles[1]) if q_styles[1] is not None else self.block2(e2)
        z = self.block3.forward_with_style(e3, q_styles[2]) if q_styles[2] is not None else self.block3(e3)
        return z, [e1, e2, e3]


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
        self.blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        for _ in range(num_skips):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, hid_ch))
            in_ch = hid_ch

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
            nn.Tanh(),
        )

    def forward(self, z, skips):
        x = z
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(x, skip)
        x = self.refine(x)
        return self.head(x)


@torch.no_grad()
def spatial_q_replace(q_content, q_style, z_content, z_style, pool_size=64):
    B, C, H, W = q_content.shape
    # assert B == 1 and q_style.shape[0] == 1

    zc = z_content.unsqueeze(0) if z_content.dim() == 3 else z_content
    zs = z_style.unsqueeze(0) if z_style.dim() == 3 else z_style

    zc_pool = F.adaptive_avg_pool2d(zc, (pool_size, pool_size))
    zs_pool = F.adaptive_avg_pool2d(zs, (pool_size, pool_size))

    fc = zc_pool.view(C, -1).T
    fs = zs_pool.view(C, -1).T
    fc_norm = F.normalize(fc, p=2, dim=1)
    fs_norm = F.normalize(fs, p=2, dim=1)

    sim = torch.matmul(fc_norm, fs_norm.T)
    mapping_idx = sim.argmax(dim=1)

    qs_pool = F.adaptive_avg_pool2d(q_style, (pool_size, pool_size))
    qs_flat = qs_pool.view(1, C, -1)

    mapping_idx = mapping_idx.view(1, 1, -1).expand(1, C, -1)
    q_mapped_flat = torch.gather(qs_flat, dim=2, index=mapping_idx)

    q_mapped_pool = q_mapped_flat.view(1, C, pool_size, pool_size)
    q_mapped = F.interpolate(q_mapped_pool, size=(H, W), mode='bilinear', align_corners=False)

    return q_mapped


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
    print(f"Device: {device}, Seed: {SEED}, Num per style: {NUM_PER_STYLE}, Batch: {BATCH_SIZE}")

    out_dir = Path("ed_benchmark")
    out_dir.mkdir(exist_ok=True)

    coco_root = r"data\coco\train2017"
    wiki_root = r"data\wikiart_images"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    encoder_ckpt = Path("checkpoints") / "encoder.pth"
    decoder_ckpt = Path("checkpoints") / "decoder.pth"
    if not encoder_ckpt.exists() or not decoder_ckpt.exists():
        raise FileNotFoundError("Encoder or decoder checkpoint not found")

    win_size = 4
    encoder = TextureSegmentor(dim=18, window_size=win_size).to(device)
    decoder = Decoder(bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3).to(device)

    ckp = torch.load(encoder_ckpt, map_location=device)
    encoder.load_state_dict(ckp["model_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    ckp = torch.load(decoder_ckpt, map_location=device)
    decoder.load_state_dict(ckp["model_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
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

        # 加载风格图（单张，共享）
        style_path = style_files[0]
        style_tensor = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

        # 加载内容图（30张）
        content_tensors = []
        for i in range(NUM_PER_STYLE):
            cp = content_batch[i]
            content_tensors.append(transform(Image.open(cp).convert("RGB")).unsqueeze(0).to(device))

        # 计算 direct_loss（单张，不计时）
        direct_losses = []
        for ct in content_tensors:
            direct_losses.append(vgg_style(ct, style_tensor).item())

        # 风格图特征（只算一次，不计时）
        with torch.no_grad():
            z_b, skips_b = encoder.forward_with_skips(style_tensor)
            e1_b = encoder.stem(style_tensor)
            q1_b = encoder.block1.get_q(e1_b)
            e2_b = encoder.block1(e1_b)
            q2_b = encoder.block2.get_q(e2_b)
            e3_b = encoder.block2(e2_b)
            q3_b = encoder.block3.get_q(e3_b)

        # Batch 推理
        for b_start in range(0, NUM_PER_STYLE, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, NUM_PER_STYLE)
            actual_bs = b_end - b_start
            batch_content = torch.cat(content_tensors[b_start:b_end], dim=0)

            t_start = time.perf_counter()

            with torch.no_grad():
                z_a, skips_a = encoder.forward_with_skips(batch_content)
                e1_a = encoder.stem(batch_content)
                q1_a = encoder.block1.get_q(e1_a)
                e2_a = encoder.block1(e1_a)
                q2_a = encoder.block2.get_q(e2_a)
                e3_a = encoder.block2(e2_a)
                q3_a = encoder.block3.get_q(e3_a)

                q1_replaced = spatial_q_replace(q1_a, q1_b, z_a[0], z_b[0], pool_size=POOL_SIZE)
                q2_replaced = spatial_q_replace(q2_a, q2_b, z_a[0], z_b[0], pool_size=POOL_SIZE)
                q3_replaced = spatial_q_replace(q3_a, q3_b, z_a[0], z_b[0], pool_size=POOL_SIZE)

                q1_mixed = ALPHA_Q * q1_replaced + (1 - ALPHA_Q) * q1_a
                q2_mixed = ALPHA_Q * q2_replaced + (1 - ALPHA_Q) * q2_a
                q3_mixed = ALPHA_Q * q3_replaced + (1 - ALPHA_Q) * q3_a

                z_attn, skips_attn = encoder.forward_with_skips_and_style(
                    batch_content, [q1_mixed, q2_mixed, q3_mixed]
                )
                rec_styled = decoder(z_attn, skips_attn)

            t_end = time.perf_counter()
            per_img_sec = (t_end - t_start) / actual_bs

            # 逐张计算 stylized_loss
            for j in range(actual_bs):
                idx = b_start + j
                sty_loss = vgg_style(rec_styled[j:j+1], style_tensor).item()
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        class_name,
                        fmt4(per_img_sec),
                        fmt4(direct_losses[idx]),
                        fmt4(sty_loss)
                    ])

            print(f"  Batch [{b_start:02d}-{b_end:02d}] PerImg={fmt4(per_img_sec)}s")

    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()