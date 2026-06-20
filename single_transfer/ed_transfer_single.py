import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models

# ========================== 配置 ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = Path("checkpoints")
COCO_ROOT = r"data\coco\train2017"
WIKI_ROOT = r"data\wikiart_images"
OUT_IMG = Path("ed_visualization.png")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_SIZE = 18

# ========================== 工具函数 ==========================
def list_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    root = Path(root)
    files = [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)

def get_wikiart_classes(wiki_root):
    classes = {}
    root = Path(wiki_root)
    for split in ["train", "test"]:
        sp = root / split
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            files = [str(f) for f in d.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
            if files:
                classes.setdefault(d.name, []).extend(files)
    for k in classes:
        classes[k] = sorted(set(classes[k]))
    return classes

def tensor_to_pil(img_tensor):
    img = img_tensor.squeeze(0).cpu().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def add_label(pil_img, text, font_path=FONT_PATH, font_size=FONT_SIZE):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    x, y = 8, 8
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=(255,255,255))
    return pil_img

# ========================== 模型定义 ==========================
class MultiScaleStem(nn.Module):
    def __init__(self, out_dim=18):
        super().__init__()
        assert out_dim == 18
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

def window_partition_overlap(x, window_size, stride):
    B, C, H, W = x.shape
    ws = window_size
    x = F.unfold(x, kernel_size=ws, stride=stride)
    L = x.shape[-1]
    x = x.view(B, C, ws*ws, L)
    x = x.permute(0, 3, 2, 1).contiguous()
    return x, stride

def window_reverse_overlap(x, B, C, H, W, window_size, stride):
    ws = window_size
    x = x.permute(0, 3, 2, 1).contiguous()
    x = x.view(B, C*ws*ws, -1)
    out = F.fold(x, output_size=(H, W), kernel_size=ws, stride=stride)
    ones = torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype)
    ones_unfold = F.unfold(ones, kernel_size=ws, stride=stride)
    norm = F.fold(ones_unfold, output_size=(H, W), kernel_size=ws, stride=stride)
    return out / (norm + 1e-8)

class CNNTransformerBlock(nn.Module):
    def __init__(self, dim=18, window_size=4):
        super().__init__()
        self.dim = dim
        self.win = window_size
        self.shift = window_size // 2
        self.stride = window_size // 2
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False), nn.BatchNorm2d(dim),
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
    def forward_with_style(self, x, Q_style):
        local = self.local(x)
        B1 = self._window_attn(Q_style)
        Q_shifted = torch.roll(Q_style, shifts=(self.shift, self.shift), dims=(2, 3))
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
    def forward(self, x):
        z = self.stem(x); z = self.block1(z); z = self.block2(z); z = self.block3(z)
        return z
    @torch.no_grad()
    def forward_with_skips(self, x):
        e1 = self.stem(x); e2 = self.block1(e1); e3 = self.block2(e2); z = self.block3(e3)
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
            nn.Conv2d(in_ch+skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        return self.conv(torch.cat([x, skip], dim=1))

class Decoder(nn.Module):
    def __init__(self, bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        for _ in range(num_skips):
            self.blocks.append(DecoderBlock(in_ch, skip_ch, hid_ch))
            in_ch = hid_ch
        self.refine = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(hid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(hid_ch), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1, bias=True), nn.Tanh(),
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
    assert B == 1 and q_style.shape[0] == 1
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
    return F.interpolate(q_mapped_pool, size=(H, W), mode='bilinear', align_corners=False)

# ========================== VGG 风格损失 ==========================
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

# ========================== 主流程 ==========================
def main():
    print(f"Device: {DEVICE}")

    coco_imgs = list_images(COCO_ROOT)
    wiki_classes = get_wikiart_classes(WIKI_ROOT)
    assert coco_imgs and wiki_classes, "数据集路径为空，请检查 COCO_ROOT / WIKI_ROOT"

    # 真正随机选图（不固定种子）
    content_path = random.choice(coco_imgs)
    style_class = random.choice(list(wiki_classes.keys()))
    style_path = random.choice(wiki_classes[style_class])

    print(f"内容图: {content_path}")
    print(f"风格图 ({style_class}): {style_path}")

    ct = TRANSFORM(Image.open(content_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    st = TRANSFORM(Image.open(style_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    encoder = TextureSegmentor(dim=18, window_size=4).to(DEVICE)
    ckp_e = torch.load(CKPT_DIR / "encoder.pth", map_location=DEVICE)
    encoder.load_state_dict(ckp_e["model_state_dict"])
    encoder.eval()

    decoder = Decoder(bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3).to(DEVICE)
    ckp_d = torch.load(CKPT_DIR / "decoder.pth", map_location=DEVICE)
    decoder.load_state_dict(ckp_d["model_state_dict"])
    decoder.eval()

    with torch.no_grad():
        z_a, skips_a = encoder.forward_with_skips(ct)
        rec = decoder(z_a, skips_a)

        z_b, skips_b = encoder.forward_with_skips(st)
        e1_b = encoder.stem(st); q1_b = encoder.block1.get_q(e1_b)
        e2_b = encoder.block1(e1_b); q2_b = encoder.block2.get_q(e2_b)
        e3_b = encoder.block2(e2_b); q3_b = encoder.block3.get_q(e3_b)

        e1_a = encoder.stem(ct); q1_a = encoder.block1.get_q(e1_a)
        e2_a = encoder.block1(e1_a); q2_a = encoder.block2.get_q(e2_a)
        e3_a = encoder.block2(e2_a); q3_a = encoder.block3.get_q(e3_a)

        q1_r = spatial_q_replace(q1_a, q1_b, z_a[0], z_b[0], pool_size=64)
        q2_r = spatial_q_replace(q2_a, q2_b, z_a[0], z_b[0], pool_size=64)
        q3_r = spatial_q_replace(q3_a, q3_b, z_a[0], z_b[0], pool_size=64)

        q1_m = 1.0 * q1_r + 0.0 * q1_a
        q2_m = 1.0 * q2_r + 0.0 * q2_a
        q3_m = 1.0 * q3_r + 0.0 * q3_a

        z_attn, skips_attn = encoder.forward_with_skips_and_style(ct, [q1_m, q2_m, q3_m])
        stylized = decoder(z_attn, skips_attn)

        vgg_path = CKPT_DIR / "vgg.pth"
        if vgg_path.exists():
            vgg_loss = VGGStyleLoss(str(vgg_path)).to(DEVICE)
            print(f"重建图风格损失: {vgg_loss(rec, st).item():.4f}")
            print(f"风格迁移图损失: {vgg_loss(stylized, st).item():.4f}")

    pil_content  = add_label(tensor_to_pil(ct),  "内容图")
    pil_style    = add_label(tensor_to_pil(st),  "风格图")
    pil_rec      = add_label(tensor_to_pil(rec), "重建图")
    pil_stylized = add_label(tensor_to_pil(stylized), "风格迁移图")

    W, H = pil_content.size
    canvas = Image.new("RGB", (W * 4, H), (0, 0, 0))
    canvas.paste(pil_content,  (0 * W, 0))
    canvas.paste(pil_style,    (1 * W, 0))
    canvas.paste(pil_rec,       (2 * W, 0))
    canvas.paste(pil_stylized, (3 * W, 0))

    canvas.save(OUT_IMG)
    print(f"\n结果已保存: {OUT_IMG.absolute()}")

if __name__ == "__main__":
    main()