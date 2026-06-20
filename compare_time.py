# compare_50_methods.py
import os
import time
import random
import csv
from pathlib import Path
from glob import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================== 配置 ==========================
SEED = 42
NUM_SAMPLES = 50
FM_BATCH_SIZE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# 路径
COCO_ROOT = r"data\coco\train2017"
WIKI_ROOT = r"data\wikiart_images"
CKPT_DIR = Path("checkpoints")

# ========================== 工具 ==========================
def list_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    root = Path(root)
    files = []
    if root.is_dir():
        for p in root.rglob("*"):
            if p.suffix.lower() in exts:
                files.append(str(p))
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


# ========================== 1. 流匹配模型 ==========================
C0, C1, C2, C3 = 512, 1024, 2048, 4096


class DSConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.gn = nn.GroupNorm(min(32, c_out), c_out)
    def forward(self, x):
        return self.gn(self.pw(self.dw(x)))


class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, dim))
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        dev = t.device
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=dev)) * torch.arange(half, device=dev) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class GAB_TypeA(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), DSConv(4, C0, 3, 1, 1))
        self.c_up = nn.Sequential(nn.Upsample(scale_factor=4, mode="nearest"), DSConv(256, 64, 3, 1, 1))
        self.expand = DSConv(64, C0, 1, 1, 0)
        self.out_gn = nn.GroupNorm(min(32, C0), C0)
    def forward(self, z_raw, t_emb, cond_s_raw, cond_c_raw):
        z = self.z_up(z_raw) + t_emb
        v = self.c_up(cond_c_raw)
        q = cond_s_raw
        B, _, H, W = q.shape
        N = H * W
        qf = q.view(B, 64, N)
        vf = v.view(B, 64, N)
        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N ** 0.5)
        gram = F.softmax(gram + 1e-6, dim=-1)
        out = torch.bmm(gram, vf).view(B, 64, H, W)
        out = self.expand(out)
        return F.silu(self.out_gn(out + z)), q, v


class GABlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, has_qv_skip=False):
        super().__init__()
        self.has_qv_skip = has_qv_skip
        self.stride = stride
        self.compress = DSConv(c_in, c_out, 3, stride, 1) if stride != 1 else DSConv(c_in, c_out, 1, 1, 0)
        cur_c = 64 // 2 if has_qv_skip else 64
        self.split_q = DSConv(c_out, cur_c, 1, 1, 0)
        self.split_v = DSConv(c_out, cur_c, 1, 1, 0)
        if has_qv_skip:
            self.skip_q = DSConv(64, cur_c, 1, 1, 0)
            self.skip_v = DSConv(64, cur_c, 1, 1, 0)
        self.expand = DSConv(64, c_out, 1, 1, 0)
        self.res = DSConv(c_in, c_out, 3, stride, 1) if stride != 1 or c_in != c_out else None
        self.out_gn = nn.GroupNorm(min(32, c_out), c_out)
    def forward(self, x, t_emb, q_prev=None, v_prev=None):
        x_comp = self.compress(x)
        q_cur = self.split_q(x_comp + t_emb)
        v_cur = self.split_v(x_comp + t_emb)
        if self.has_qv_skip and q_prev is not None:
            if q_prev.shape[-2:] != q_cur.shape[-2:]:
                q_prev = F.interpolate(q_prev, size=q_cur.shape[-2:], mode="nearest")
                v_prev = F.interpolate(v_prev, size=v_cur.shape[-2:], mode="nearest")
            q_prev = self.skip_q(q_prev)
            v_prev = self.skip_v(v_prev)
            q = torch.cat([q_cur, q_prev], dim=1)
            v = torch.cat([v_cur, v_prev], dim=1)
        else:
            q, v = q_cur, v_cur
        B, C, H, W = q.shape
        N = H * W
        qf = q.view(B, C, N)
        vf = v.view(B, C, N)
        gram = torch.bmm(qf, qf.transpose(1, 2)) / (N ** 0.5)
        gram = F.softmax(gram + 1e-6, dim=-1)
        out = torch.bmm(gram, vf).view(B, C, H, W)
        out = self.expand(out)
        res = self.res(x) if self.res is not None else x
        return F.silu(self.out_gn(out + res)), q, v


class ResNetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        rn = models.resnet18(weights=None)
        self.conv1 = rn.conv1; self.bn1 = rn.bn1; self.relu = rn.relu
        self.maxpool = rn.maxpool; self.layer1 = rn.layer1; self.layer2 = rn.layer2; self.layer3 = rn.layer3
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)
        x = self.layer2(f1)
        f3 = self.layer3(x)
        return f1, f3


class ScalableFlowMatchingNet(nn.Module):
    def __init__(self, depth_factor=2.0):
        super().__init__()
        self.depth_factor = depth_factor
        self.n_enc0 = max(2, int(round(2 * depth_factor)))
        self.n_enc1 = max(2, int(round(2 * depth_factor)))
        self.n_enc2 = max(4, int(round(4 * depth_factor)))
        self.n_enc3 = max(4, int(round(4 * depth_factor)))
        self.n_dec3 = max(3, int(round(3 * depth_factor)))
        self.n_dec2 = max(3, int(round(3 * depth_factor)))
        self.n_dec1 = max(3, int(round(3 * depth_factor)))
        self.n_dec0 = max(3, int(round(3 * depth_factor)))

        self.time_embed = TimeEmbed(128)
        time_dims = [C0]*self.n_enc0 + [C1]*self.n_enc1 + [C2]*self.n_enc2 + [C3]*self.n_enc3 + [C3]*self.n_dec3 + [C2]*self.n_dec2 + [C1]*self.n_dec1 + [C0]*self.n_dec0
        self.time_projs = nn.ModuleList([nn.Linear(128, d) for d in time_dims])

        self.latent_aug = DSConv(4, C0, 1, 1, 0)

        self.enc0 = nn.ModuleList([GAB_TypeA()] + [GABlock(C0, C0, has_qv_skip=True) for _ in range(1, self.n_enc0)])
        self.enc1 = nn.ModuleList([GABlock(C0, C1, stride=2, has_qv_skip=True)] + [GABlock(C1, C1, has_qv_skip=True) for _ in range(1, self.n_enc1)])
        self.enc2 = nn.ModuleList([GABlock(C1, C2, stride=2, has_qv_skip=True)] + [GABlock(C2, C2, has_qv_skip=True) for _ in range(1, self.n_enc2)])
        self.enc3 = nn.ModuleList([GABlock(C2, C3, stride=2, has_qv_skip=True)] + [GABlock(C3, C3, has_qv_skip=True) for _ in range(1, self.n_enc3)])

        self.dec3 = nn.ModuleList([GABlock(C3*2, C3, has_qv_skip=True)] + [GABlock(C3, C3, has_qv_skip=True) for _ in range(1, self.n_dec3)])
        self.dec2 = nn.ModuleList([GABlock(C3+C2, C2, has_qv_skip=True)] + [GABlock(C2, C2, has_qv_skip=True) for _ in range(1, self.n_dec2)])
        self.dec1 = nn.ModuleList([GABlock(C2+C1, C1, has_qv_skip=True)] + [GABlock(C1, C1, has_qv_skip=True) for _ in range(1, self.n_dec1)])
        self.dec0 = nn.ModuleList([GABlock(C1+C0, C0, has_qv_skip=True)] + [GABlock(C0, C0, has_qv_skip=True) for _ in range(1, self.n_dec0)])

        self.to_latent = DSConv(C0, 4, 3, 2, 1)

    def forward(self, z_raw, t, cond_s_raw, cond_c_raw):
        B = z_raw.shape[0]
        h4, w4 = z_raw.shape[2]*2, z_raw.shape[3]*2
        t_vec = self.time_embed(t)
        t_embs = [proj(t_vec)[:, :, None, None] for proj in self.time_projs]
        t_idx = 0
        def get_t_emb():
            nonlocal t_idx
            te = t_embs[t_idx]; t_idx += 1; return te

        z_aug = self.latent_aug(z_raw)
        z_aug_up = F.interpolate(z_aug, size=(h4, w4), mode="nearest")

        skips, qv_skips = {}, {}
        x, q_prev, v_prev = None, None, None
        for i, blk in enumerate(self.enc0):
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            if i == 0:
                # GAB_TypeA 需要外部条件 cond_s_raw / cond_c_raw
                x, q_prev, v_prev = blk(z_raw, t_emb, cond_s_raw, cond_c_raw)
            else:
                x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            skips["enc0"] = x
        qv_skips["enc0"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc1):
            t_emb = get_t_emb().expand(B, -1, h4//2, w4//2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc1)-1: skips["enc1"] = x
        qv_skips["enc1"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc2):
            t_emb = get_t_emb().expand(B, -1, h4//4, w4//4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc2)-1: skips["enc2"] = x
        qv_skips["enc2"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc3):
            t_emb = get_t_emb().expand(B, -1, h4//8, w4//8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc3)-1: skips["enc3"] = x
        qv_skips["enc3"] = (q_prev, v_prev)

        x = torch.cat([x, skips["enc3"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4//8, w4//8)
        x, q_prev, v_prev = self.dec3[0](x, t_emb, qv_skips["enc3"][0], qv_skips["enc3"][1])
        for blk in self.dec3[1:]:
            t_emb = get_t_emb().expand(B, -1, h4//8, w4//8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4//4, w4//4), mode="nearest")
        x = torch.cat([x, skips["enc2"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4//4, w4//4)
        x, q_prev, v_prev = self.dec2[0](x, t_emb, qv_skips["enc2"][0], qv_skips["enc2"][1])
        for blk in self.dec2[1:]:
            t_emb = get_t_emb().expand(B, -1, h4//4, w4//4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4//2, w4//2), mode="nearest")
        x = torch.cat([x, skips["enc1"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4//2, w4//2)
        x, q_prev, v_prev = self.dec1[0](x, t_emb, qv_skips["enc1"][0], qv_skips["enc1"][1])
        for blk in self.dec1[1:]:
            t_emb = get_t_emb().expand(B, -1, h4//2, w4//2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4, w4), mode="nearest")
        skip0_fused = skips["enc0"] + z_aug_up
        x = torch.cat([x, skip0_fused], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4, w4)
        x, q_prev, v_prev = self.dec0[0](x, t_emb, qv_skips["enc0"][0], qv_skips["enc0"][1])
        for blk in self.dec0[1:]:
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        return self.to_latent(x)


@torch.no_grad()
def forward_ode(model, z_data, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_data.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i*dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z - dt * v
    return z

@torch.no_grad()
def reverse_ode(model, z_noise, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_noise.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i*dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z + dt * v
    return z


# ========================== 2. Encoder-Decoder 模型 ==========================
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
        return F.relu(self.out_norm(x + local + attn))

    # ===== 新增：允许从外部传入 Q，跳过 get_q =====
    def forward_with_q(self, x, q):
        local = self.local(x)
        Q = q
        B1 = self._window_attn(Q)
        Q_shifted = torch.roll(Q, shifts=(self.shift, self.shift), dims=(2, 3))
        B2 = self._window_attn(Q_shifted)
        B2 = torch.roll(B2, shifts=(-self.shift, -self.shift), dims=(2, 3))
        attn = B1 + B2
        return F.relu(self.out_norm(x + local + attn))

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

    # ===== 新增：用外部替换后的 q_list 重新前向传播 =====
    @torch.no_grad()
    def forward_with_skips_and_style(self, x, q_list):
        e1 = self.stem(x)
        e2 = self.block1.forward_with_q(e1, q_list[0])
        e3 = self.block2.forward_with_q(e2, q_list[1])
        z = self.block3.forward_with_q(e3, q_list[2])
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


# ========================== 3. 颜色迁移模型 & 工具 ==========================
class StyleColorPredictor(nn.Module):
    def __init__(self, bins=33):
        super().__init__()
        self.bins = bins
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, bins ** 3)
        )
    def forward(self, style_img):
        x = (style_img + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.features(x).view(x.size(0), -1)
        logits = self.fc(feat)
        log_prob = F.log_softmax(logits, dim=-1)
        return log_prob.exp().view(-1, self.bins, self.bins, self.bins), log_prob

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
    out = torch.zeros(B, bins*bins*bins, device=x.device, dtype=x.dtype)
    offsets = SPLAT_OFFSETS.to(x.device).view(1, 1, 8, 3)
    idx_all = base.unsqueeze(2) + offsets
    idx_all = idx_all.clamp(0, bins - 1)
    flat_all = idx_all[..., 0]*bins*bins + idx_all[..., 1]*bins + idx_all[..., 2]
    w_all = torch.stack([
        (1-frac[...,0])*(1-frac[...,1])*(1-frac[...,2]),
        frac[...,0]*(1-frac[...,1])*(1-frac[...,2]),
        (1-frac[...,0])*frac[...,1]*(1-frac[...,2]),
        frac[...,0]*frac[...,1]*(1-frac[...,2]),
        (1-frac[...,0])*(1-frac[...,1])*frac[...,2],
        frac[...,0]*(1-frac[...,1])*frac[...,2],
        (1-frac[...,0])*frac[...,1]*frac[...,2],
        frac[...,0]*frac[...,1]*frac[...,2],
    ], dim=2)
    out = torch.scatter_add(out, 1, flat_all.reshape(B, N*8), w_all.reshape(B, N*8))
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
    for _ in range(n_iter):
        optimizer.zero_grad()
        lut = (identity + delta).clamp(0, 1)
        lut_gs = lut.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear', padding_mode='border', align_corners=True)
        out = out.squeeze(2)
        H_dist = splat_3d(out, bins).squeeze(0)
        H_safe = H_dist + 1e-8
        P_safe = P_target + 1e-8
        kl = (H_safe * (H_safe.log() - P_safe.log())).sum()
        tv_r = torch.abs(lut[1:,:,:,:] - lut[:-1,:,:,:]).mean()
        tv_g = torch.abs(lut[:,1:,:,:] - lut[:,:-1,:,:]).mean()
        tv_b = torch.abs(lut[:,:,1:,:] - lut[:,:,:-1,:]).mean()
        loss = kl + tv_weight * (tv_r + tv_g + tv_b)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        lut_final = (identity + delta).clamp(0, 1)
        lut_gs = lut_final.permute(3, 2, 1, 0).unsqueeze(0)
        out = F.grid_sample(lut_gs, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)
        return out * 2.0 - 1.0


# ========================== 4. Gatys NST ==========================
class GatysNST:
    def __init__(self, device):
        self.device = device
        self.cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.content_layer = 21
        self.style_layers = [0, 5, 10, 19, 28]
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _normalize(self, x):
        return ((x + 1.0) / 2.0 - self.mean) / self.std

    def _get_content_feature(self, x):
        out = x
        for i, layer in enumerate(self.cnn):
            out = layer(out)
            if i == self.content_layer:
                return out
        return out

    def _get_style_features(self, x):
        features = []
        out = x
        for i, layer in enumerate(self.cnn):
            out = layer(out)
            if i in self.style_layers:
                features.append(out)
        return features

    @staticmethod
    def gram_matrix(feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, -1)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def run(self, content, style, steps=500, content_weight=1.0, style_weight=1e7, lr=0.01):
        content_target = self._get_content_feature(self._normalize(content))
        style_feats = self._get_style_features(self._normalize(style))
        style_targets = [self.gram_matrix(f) for f in style_feats]

        input_img = content.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([input_img], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            inp_norm = self._normalize(input_img)
            c_feat = self._get_content_feature(inp_norm)
            s_feats = self._get_style_features(inp_norm)
            c_loss = F.mse_loss(c_feat, content_target)
            s_loss = 0.0
            for a, b in zip(s_feats, style_targets):
                s_loss += F.mse_loss(self.gram_matrix(a), b)
            loss = content_weight * c_loss + style_weight * s_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                input_img.clamp_(-1, 1)
        return input_img.detach()


# ========================== 5. 主流程 ==========================
def main():
    print(f"Device: {DEVICE}")

    # 数据
    coco_imgs = list_images(COCO_ROOT)
    wiki_classes = get_wikiart_classes(WIKI_ROOT)
    wiki_imgs = []
    for files in wiki_classes.values():
        wiki_imgs.extend(files)
    random.shuffle(coco_imgs)
    random.shuffle(wiki_imgs)
    coco_selected = coco_imgs[:NUM_SAMPLES]
    wiki_selected = wiki_imgs[:NUM_SAMPLES]

    # 变换
    transform_fm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_others = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载模型
    print("Loading models...")

    # FM 相关
    vgg_path = CKPT_DIR / "vgg.pth"
    if not vgg_path.exists():
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(vgg.state_dict(), vgg_path)
    rn = models.resnet18(weights=None)
    resnet_path = CKPT_DIR / "resnet18.pth"
    if resnet_path.exists():
        rn.load_state_dict(torch.load(resnet_path, map_location="cpu"))
    else:
        rn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        torch.save(rn.state_dict(), resnet_path)
    resnet_fm = ResNetFeat().to(DEVICE)
    resnet_fm.load_state_dict(rn.state_dict(), strict=False)
    resnet_fm.eval()
    for p in resnet_fm.parameters():
        p.requires_grad = False

    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained(str(CKPT_DIR / "taesd"), local_files_only=True).to(DEVICE)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False

    fm_model = ScalableFlowMatchingNet(depth_factor=2.0).to(DEVICE)
    fm_ckpt = torch.load(CKPT_DIR / "fm_gram.pth", map_location=DEVICE)
    fm_model.load_state_dict(fm_ckpt["model"], strict=False)
    fm_model.eval()
    for p in fm_model.parameters():
        p.requires_grad = False

    # ED 相关
    encoder = TextureSegmentor(dim=18, window_size=4).to(DEVICE)
    decoder = Decoder(bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3).to(DEVICE)
    ckp_e = torch.load(CKPT_DIR / "encoder.pth", map_location=DEVICE)
    encoder.load_state_dict(ckp_e["model_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    ckp_d = torch.load(CKPT_DIR / "decoder.pth", map_location=DEVICE)
    decoder.load_state_dict(ckp_d["model_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # Color 相关
    color_model = StyleColorPredictor(bins=33).to(DEVICE)
    color_model.load_state_dict(torch.load(CKPT_DIR / "latest.pth", map_location=DEVICE)["model"])
    color_model.eval()
    for p in color_model.parameters():
        p.requires_grad = False

    # Gatys
    gatys = GatysNST(DEVICE)

    # 预加载所有图像tensor
    print("Preloading images...")
    content_fm = [transform_fm(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE) for p in coco_selected]
    style_fm = [transform_fm(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE) for p in wiki_selected]
    content_others = [transform_others(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE) for p in coco_selected]
    style_others = [transform_others(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE) for p in wiki_selected]

    # 预提取FM特征
    print("Precomputing FM features...")
    fm_content_z, fm_content_s, fm_content_c = [], [], []
    fm_style_s, fm_style_c = [], []
    for i in range(NUM_SAMPLES):
        c01 = (content_fm[i] + 1.0) / 2.0
        s01 = (style_fm[i] + 1.0) / 2.0
        with torch.no_grad():
            cs, cc = resnet_fm(c01)
            z = taesd.encode(c01).latents
            ss, sc = resnet_fm(s01)
        fm_content_z.append(z)
        fm_content_s.append(cs)
        fm_content_c.append(cc)
        fm_style_s.append(ss)
        fm_style_c.append(sc)

    # 结果容器
    fm_single_times = [0.0] * NUM_SAMPLES
    fm_batch_times = [0.0] * NUM_SAMPLES
    ed_times = []
    color_times = []
    gatys_times = []

    # ---------- 1. 流匹配 单张 (batch_size=1) ----------
    print("\nRunning Flow Matching (single, batch=1)...")
    for i in range(NUM_SAMPLES):
        z = fm_content_z[i]
        cs = fm_content_s[i]
        cc = fm_content_c[i]
        ss = fm_style_s[i]

        t0 = time.perf_counter()
        with torch.no_grad():
            z_noise = forward_ode(fm_model, z, cs, cc, n_steps=20)
            cond_s_mixed = 1.0 * ss + 0.0 * cs
            z_stylized = reverse_ode(fm_model, z_noise, cond_s_mixed, cc, n_steps=20)
            _ = taesd.decode(z_stylized).sample
        t1 = time.perf_counter()
        fm_single_times[i] = t1 - t0

    # ---------- 2. 流匹配 批量 (batch_size=10) ----------
    print("Running Flow Matching (batch=10)...")
    for b_start in range(0, NUM_SAMPLES, FM_BATCH_SIZE):
        b_end = min(b_start + FM_BATCH_SIZE, NUM_SAMPLES)
        actual_bs = b_end - b_start
        z_batch = torch.cat([fm_content_z[i] for i in range(b_start, b_end)], dim=0)
        cs_batch = torch.cat([fm_content_s[i] for i in range(b_start, b_end)], dim=0)
        cc_batch = torch.cat([fm_content_c[i] for i in range(b_start, b_end)], dim=0)
        ss_batch = torch.cat([fm_style_s[i] for i in range(b_start, b_end)], dim=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            z_noise = forward_ode(fm_model, z_batch, cs_batch, cc_batch, n_steps=20)
            cond_s_mixed = 1.0 * ss_batch + 0.0 * cs_batch
            z_stylized = reverse_ode(fm_model, z_noise, cond_s_mixed, cc_batch, n_steps=20)
            _ = taesd.decode(z_stylized).sample
        t1 = time.perf_counter()
        per_img = (t1 - t0) / actual_bs
        for i in range(b_start, b_end):
            fm_batch_times[i] = per_img

    # ---------- 3. 单次迁移 (Encoder-Decoder) ----------
    print("Running Encoder-Decoder...")
    for i in range(NUM_SAMPLES):
        ct = content_others[i]
        st = style_others[i]
        t0 = time.perf_counter()
        with torch.no_grad():
            z_a, skips_a = encoder.forward_with_skips(ct)
            z_b, skips_b = encoder.forward_with_skips(st)
            e1_b = encoder.stem(st)
            q1_b = encoder.block1.get_q(e1_b)
            e2_b = encoder.block1(e1_b)
            q2_b = encoder.block2.get_q(e2_b)
            e3_b = encoder.block2(e2_b)
            q3_b = encoder.block3.get_q(e3_b)
            e1_a = encoder.stem(ct)
            q1_a = encoder.block1.get_q(e1_a)
            e2_a = encoder.block1(e1_a)
            q2_a = encoder.block2.get_q(e2_a)
            e3_a = encoder.block2(e2_a)
            q3_a = encoder.block3.get_q(e3_a)
            q1_r = spatial_q_replace(q1_a, q1_b, z_a[0], z_b[0], pool_size=64)
            q2_r = spatial_q_replace(q2_a, q2_b, z_a[0], z_b[0], pool_size=64)
            q3_r = spatial_q_replace(q3_a, q3_b, z_a[0], z_b[0], pool_size=64)
            z_attn, skips_attn = encoder.forward_with_skips_and_style(ct, [q1_r, q2_r, q3_r])
            _ = decoder(z_attn, skips_attn)
        t1 = time.perf_counter()
        ed_times.append(t1 - t0)

    # ---------- 4. Adam优化 (颜色迁移 3D LUT) ----------
    print("Running Color LUT (Adam optimization)...")
    for i in range(NUM_SAMPLES):
        ct = content_others[i]
        st = style_others[i]
        with torch.no_grad():
            prob, _ = color_model(st)
        t0 = time.perf_counter()
        _ = transfer_color_lut(ct, prob[0], st, bins=33, n_iter=800, lr=0.001, tv_weight=10)
        t1 = time.perf_counter()
        color_times.append(t1 - t0)

    # ---------- 5. Gatys Gram矩阵 ----------
    print("Running Gatys (Gram-based)...")
    for i in range(NUM_SAMPLES):
        ct = content_others[i]
        st = style_others[i]
        t0 = time.perf_counter()
        _ = gatys.run(ct, st, steps=500, content_weight=1.0, style_weight=1e7, lr=0.01)
        t1 = time.perf_counter()
        gatys_times.append(t1 - t0)

    # 保存结果
    out_path = Path("compare_50_times.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flow_matching_single", "flow_matching_batch", "single_transfer", "adam_opt", "gatys"])
        for i in range(NUM_SAMPLES):
            writer.writerow([
                f"{fm_single_times[i]:.4g}",
                f"{fm_batch_times[i]:.4g}",
                f"{ed_times[i]:.4g}",
                f"{color_times[i]:.4g}",
                f"{gatys_times[i]:.4g}",
            ])

    print(f"\nDone. Saved to {out_path}")
    print(f"Flow Matching (single):    {sum(fm_single_times)/len(fm_single_times):.4g}s")
    print(f"Flow Matching (batch=10):  {sum(fm_batch_times)/len(fm_batch_times):.4g}s")
    print(f"Encoder-Decoder (single):  {sum(ed_times)/len(ed_times):.4g}s")
    print(f"Color LUT (Adam 800it):    {sum(color_times)/len(color_times):.4g}s")
    print(f"Gatys (Gram 500it):        {sum(gatys_times)/len(gatys_times):.4g}s")


if __name__ == "__main__":
    main()