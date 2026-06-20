# compare_100_fm_vs_gatys.py
import os
import time
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from diffusers import AutoencoderTiny
from torchvision.models import ResNet18_Weights

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================== 配置 ==========================
N_STEPS = 20
DEPTH_FACTOR = 2
LAMBDA = 0.5
FM_CKPT_NAME = 'fm_gram.pth'
NUM_PAIRS = 100

CACHE_DIR = Path('cache_gram')
COCO_CACHE = CACHE_DIR / 'coco_files.pkl'
WIKI_ALL_CACHE = CACHE_DIR / 'wiki_all.pkl'
FEAT_CACHE_DIR = CACHE_DIR / 'features'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ========================== 1. Flow Matching 模型定义（与你第二份代码严格一致） ==========================

C0 = 512
C1 = 1024
C2 = 2048
C3 = 4096


def make_divisible(v, divisor=32):
    return max(divisor, int(v + divisor // 2) // divisor * divisor)


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
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=dev))
            * torch.arange(half, device=dev)
            / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class GAB_TypeA(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), DSConv(4, C0, 3, 1, 1)
        )
        self.c_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"), DSConv(256, 64, 3, 1, 1)
        )
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

        if stride != 1:
            self.compress = DSConv(c_in, c_out, 3, stride, 1)
        else:
            self.compress = DSConv(c_in, c_out, 1, 1, 0)

        cur_c = 64 // 2 if has_qv_skip else 64
        self.split_q = DSConv(c_out, cur_c, 1, 1, 0)
        self.split_v = DSConv(c_out, cur_c, 1, 1, 0)

        if has_qv_skip:
            self.skip_q = DSConv(64, cur_c, 1, 1, 0)
            self.skip_v = DSConv(64, cur_c, 1, 1, 0)

        self.expand = DSConv(64, c_out, 1, 1, 0)

        if stride != 1 or c_in != c_out:
            self.res = DSConv(c_in, c_out, 3, stride, 1)
        else:
            self.res = None
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
            q = q_cur
            v = v_cur

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
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3

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

        time_dims = []
        time_dims += [C0] * self.n_enc0
        time_dims += [C1] * self.n_enc1
        time_dims += [C2] * self.n_enc2
        time_dims += [C3] * self.n_enc3
        time_dims += [C3] * self.n_dec3
        time_dims += [C2] * self.n_dec2
        time_dims += [C1] * self.n_dec1
        time_dims += [C0] * self.n_dec0
        self.time_projs = nn.ModuleList([nn.Linear(128, d) for d in time_dims])

        self.latent_aug = DSConv(4, C0, 1, 1, 0)

        self.enc0 = nn.ModuleList()
        self.enc0.append(GAB_TypeA())
        for _ in range(1, self.n_enc0):
            self.enc0.append(GABlock(C0, C0, has_qv_skip=True))

        self.enc1 = nn.ModuleList()
        self.enc1.append(GABlock(C0, C1, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc1):
            self.enc1.append(GABlock(C1, C1, has_qv_skip=True))

        self.enc2 = nn.ModuleList()
        self.enc2.append(GABlock(C1, C2, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc2):
            self.enc2.append(GABlock(C2, C2, has_qv_skip=True))

        self.enc3 = nn.ModuleList()
        self.enc3.append(GABlock(C2, C3, stride=2, has_qv_skip=True))
        for _ in range(1, self.n_enc3):
            self.enc3.append(GABlock(C3, C3, has_qv_skip=True))

        self.dec3 = nn.ModuleList()
        self.dec3.append(GABlock(C3 * 2, C3, has_qv_skip=True))
        for _ in range(1, self.n_dec3):
            self.dec3.append(GABlock(C3, C3, has_qv_skip=True))

        self.dec2 = nn.ModuleList()
        self.dec2.append(GABlock(C3 + C2, C2, has_qv_skip=True))
        for _ in range(1, self.n_dec2):
            self.dec2.append(GABlock(C2, C2, has_qv_skip=True))

        self.dec1 = nn.ModuleList()
        self.dec1.append(GABlock(C2 + C1, C1, has_qv_skip=True))
        for _ in range(1, self.n_dec1):
            self.dec1.append(GABlock(C1, C1, has_qv_skip=True))

        self.dec0 = nn.ModuleList()
        self.dec0.append(GABlock(C1 + C0, C0, has_qv_skip=True))
        for _ in range(1, self.n_dec0):
            self.dec0.append(GABlock(C0, C0, has_qv_skip=True))

        self.to_latent = DSConv(C0, 4, 3, 2, 1)

    def forward(self, z_raw, t, cond_s_raw, cond_c_raw):
        B = z_raw.shape[0]
        h4, w4 = z_raw.shape[2] * 2, z_raw.shape[3] * 2

        t_vec = self.time_embed(t)
        t_embs = [proj(t_vec)[:, :, None, None] for proj in self.time_projs]

        t_idx = 0

        def get_t_emb():
            nonlocal t_idx
            te = t_embs[t_idx]
            t_idx += 1
            return te

        z_aug = self.latent_aug(z_raw)
        z_aug_up = F.interpolate(z_aug, size=(h4, w4), mode="nearest")

        skips = {}
        qv_skips = {}

        for i, blk in enumerate(self.enc0):
            t_emb = get_t_emb().expand(B, -1, h4, w4)
            if i == 0:
                x, q_prev, v_prev = blk(z_raw, t_emb, cond_s_raw, cond_c_raw)
            else:
                x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            skips["enc0"] = x
        qv_skips["enc0"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc1):
            t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc1) - 1:
                skips["enc1"] = x
        qv_skips["enc1"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc2):
            t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc2) - 1:
                skips["enc2"] = x
        qv_skips["enc2"] = (q_prev, v_prev)

        for i, blk in enumerate(self.enc3):
            t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)
            if i == len(self.enc3) - 1:
                skips["enc3"] = x
        qv_skips["enc3"] = (q_prev, v_prev)

        x = torch.cat([x, skips["enc3"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
        x, q_prev, v_prev = self.dec3[0](x, t_emb, qv_skips["enc3"][0], qv_skips["enc3"][1])
        for blk in self.dec3[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 8, w4 // 8)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 4, w4 // 4), mode="nearest")
        x = torch.cat([x, skips["enc2"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
        x, q_prev, v_prev = self.dec2[0](x, t_emb, qv_skips["enc2"][0], qv_skips["enc2"][1])
        for blk in self.dec2[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 4, w4 // 4)
            x, q_prev, v_prev = blk(x, t_emb, q_prev, v_prev)

        x = F.interpolate(x, size=(h4 // 2, w4 // 2), mode="nearest")
        x = torch.cat([x, skips["enc1"]], dim=1)
        t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
        x, q_prev, v_prev = self.dec1[0](x, t_emb, qv_skips["enc1"][0], qv_skips["enc1"][1])
        for blk in self.dec1[1:]:
            t_emb = get_t_emb().expand(B, -1, h4 // 2, w4 // 2)
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


# ========================== 2. Gatys Gram-based（与你第三份代码严格一致） ==========================

class GatysNST:
    def __init__(self, device):
        self.device = device
        self.cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.content_layer = 21
        self.style_layers = [0, 5, 10, 19, 28]

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _normalize(self, x):
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

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


# ========================== 3. 颜色分布 KL（来自你第三份代码） ==========================

SPLAT_OFFSETS = torch.tensor([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0],
    [0,0,1], [1,0,1], [0,1,1], [1,1,1]
], dtype=torch.long)


def splat_3d(x, bins=33):
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


def compute_kl_between(img_a_01, img_b_01, bins=33):
    H_a = splat_3d(img_a_01, bins).squeeze(0) + 1e-8
    H_b = splat_3d(img_b_01, bins).squeeze(0) + 1e-8
    return (H_a * (H_a.log() - H_b.log())).sum().item()


# ========================== 4. 数据工具（来自你第二份代码） ==========================

import pickle
from glob import glob


def fast_list_images(root, cache_file=None, exts=None):
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


def get_wikiart_all(wiki_root, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return data
    wiki_root = Path(wiki_root)
    files = []
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    for split in ['train', 'test']:
        sp = wiki_root / split
        if not sp.exists():
            continue
        for p in sp.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))
    files = sorted(set(files))
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    return files


@torch.no_grad()
def get_or_cache_features(img_path, resnet, taesd, transform, device, cache_dir=FEAT_CACHE_DIR):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mtime = os.path.getmtime(img_path)
    key = f"{Path(img_path).stem}_{mtime:.0f}_gram.pt"
    cache_file = cache_dir / key
    if cache_file.exists():
        data = torch.load(cache_file, map_location=device)
        return data['tensor_01'], data['cond_s_raw'], data['cond_c_raw'], data['latent']

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    tensor_01 = (tensor + 1.0) / 2.0

    cond_s_raw, cond_c_raw = resnet(tensor_01)
    latent = taesd.encode(tensor_01).latents

    torch.save({
        'tensor_01': tensor_01.cpu(),
        'cond_s_raw': cond_s_raw.cpu(),
        'cond_c_raw': cond_c_raw.cpu(),
        'latent': latent.cpu(),
    }, cache_file)

    return tensor_01, cond_s_raw, cond_c_raw, latent


# ========================== 5. ODE 采样（来自你第二份代码） ==========================

@torch.no_grad()
def forward_ode(model, z_data, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_data.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z - dt * v
    return z


@torch.no_grad()
def reverse_ode(model, z_noise, cond_s, cond_c, n_steps=20):
    dt = 1.0 / n_steps
    z = z_noise.clone()
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device, dtype=z.dtype)
        v = model(z, t, cond_s, cond_c)
        z = z + dt * v
    return z


# ========================== 6. 可视化（1×4 横向无白边） ==========================

def save_comparison_figure(content, style, fm, gram, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    tensors = [
        ((content.detach() + 1.0) / 2.0).clamp(0, 1)[0],
        ((style.detach() + 1.0) / 2.0).clamp(0, 1)[0],
        ((fm.detach() + 1.0) / 2.0).clamp(0, 1)[0],
        ((gram.detach() + 1.0) / 2.0).clamp(0, 1)[0],
    ]
    labels = [
        'Content (COCO)',
        'Style (WikiArt)',
        'Ours (Flow Matching)',
        'Gram-based\n(w=1e7, 500it)',
    ]

    for ax, tensor, label in zip(axes, tensors, labels):
        ax.imshow(tensor.permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=11,
                color='yellow', fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    print(f"Saved {save_path}")


# ========================== 7. 主流程：100 张对比 ==========================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(time.time())
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Device: {device}, Seed: {seed}")
    print(f"Comparing {NUM_PAIRS} random pairs: Flow Matching vs Gram-based NST")

    out_dir = Path("compare_100_fm_vs_gram")
    out_dir.mkdir(exist_ok=True)

    # ---------- ResNet18 ----------
    ckpt_dir = Path('checkpoints')
    resnet_path = ckpt_dir / 'resnet18.pth'
    rn = models.resnet18(weights=None)
    if resnet_path.exists():
        rn.load_state_dict(torch.load(resnet_path, map_location='cpu'))
    else:
        rn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        torch.save(rn.state_dict(), resnet_path)
    resnet = ResNetFeat().to(device)
    resnet.load_state_dict(rn.state_dict(), strict=False)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    # ---------- TAESD ----------
    taesd_dir = ckpt_dir / 'taesd'
    taesd = AutoencoderTiny.from_pretrained(str(taesd_dir), local_files_only=True)
    taesd = taesd.to(device)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False

    # ---------- Flow Matching ----------
    model = ScalableFlowMatchingNet(depth_factor=DEPTH_FACTOR).to(device)
    fm_ckpt_path = ckpt_dir / FM_CKPT_NAME
    if not fm_ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {fm_ckpt_path}')
    ckpt = torch.load(fm_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f'Loaded ScalableFlowMatchingNet from epoch {ckpt.get("epoch", "?")}')

    # ---------- Gatys ----------
    gatys = GatysNST(device)

    # ---------- 数据 ----------
    coco_root = r'data\coco\train2017'
    wiki_root = r'data\wikiart_images'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    coco_imgs = fast_list_images(coco_root, cache_file=str(COCO_CACHE))
    wiki_imgs = get_wikiart_all(wiki_root, cache_file=str(WIKI_ALL_CACHE))

    if len(coco_imgs) < NUM_PAIRS or len(wiki_imgs) < NUM_PAIRS:
        raise RuntimeError(f"Need {NUM_PAIRS} images: coco={len(coco_imgs)}, wiki={len(wiki_imgs)}")

    random.shuffle(coco_imgs)
    random.shuffle(wiki_imgs)
    coco_selected = coco_imgs[:NUM_PAIRS]
    wiki_selected = wiki_imgs[:NUM_PAIRS]

    # 统计
    time_fm = []
    time_gram = []
    kl_fm = []
    kl_gram = []

    for i in range(NUM_PAIRS):
        print(f"\n[{i+1:03d}/{NUM_PAIRS}]")
        content_path = coco_selected[i]
        style_path = wiki_selected[i]
        print(f"  Content: {Path(content_path).name}")
        print(f"  Style:   {Path(style_path).name}")

        content = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).to(device)
        style = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(device)

        content_01 = (content + 1.0) / 2.0

        # ---------- Flow Matching ----------
        t0 = time.time()

        with torch.no_grad():
            content_01_fm, cond_s_c, cond_c_c, z_content = get_or_cache_features(
                content_path, resnet, taesd, transform, device
            )
            style_01_fm, cond_s_s, _, _ = get_or_cache_features(
                style_path, resnet, taesd, transform, device
            )

            z_noise = forward_ode(model, z_content, cond_s_c, cond_c_c, n_steps=N_STEPS)

            cond_s_mixed = LAMBDA * cond_s_s + (1.0 - LAMBDA) * cond_s_c
            z_stylized = reverse_ode(model, z_noise, cond_s_mixed, cond_c_c, n_steps=N_STEPS)
            fm_out = taesd.decode(z_stylized).sample  # [0,1]

        t1 = time.time()
        fm_sec = t1 - t0
        time_fm.append(fm_sec)

        fm_01 = fm_out
        kl_fm.append(compute_kl_between(fm_01, content_01, bins=33))

        # ---------- Gram-based ----------
        t0 = time.time()
        gram_out = gatys.run(content, style, steps=500, content_weight=1.0, style_weight=1e7, lr=0.01)
        t1 = time.time()
        gram_sec = t1 - t0
        time_gram.append(gram_sec)

        gram_01 = (gram_out + 1.0) / 2.0
        kl_gram.append(compute_kl_between(gram_01, content_01, bins=33))

        print(f"  [FM]    Time: {fm_sec:.2f}s | KL(vs Content): {kl_fm[-1]:.4f}")
        print(f"  [Gram]  Time: {gram_sec:.2f}s | KL(vs Content): {kl_gram[-1]:.4f}")

        # 保存 1×4 对比图（统一传入 [-1,1] 给可视化函数）
        save_comparison_figure(
            content, style,
            fm_out * 2.0 - 1.0,   # 转回 [-1,1] 以便可视化统一处理
            gram_out,
            out_dir / f"{i+1:03d}.png"
        )

    # ==================== 统计输出 ====================
    print(f"\n{'='*60}")
    print(f"Summary over {NUM_PAIRS} random pairs")
    print(f"{'='*60}")
    print(f"FM    - Mean Time: {statistics.mean(time_fm):.2f}s  "
          f"Mean KL(vs Content): {statistics.mean(kl_fm):.4f}")
    print(f"Gram  - Mean Time: {statistics.mean(time_gram):.2f}s  "
          f"Mean KL(vs Content): {statistics.mean(kl_gram):.4f}")
    print(f"{'='*60}")

    # 柱状图：时间
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Ours (FM)', 'Gram-based']
    means = [statistics.mean(time_fm), statistics.mean(time_gram)]
    stds = [statistics.stdev(time_fm) if len(time_fm) > 1 else 0,
            statistics.stdev(time_gram) if len(time_gram) > 1 else 0]
    colors = ['#e74c3c', '#3498db']
    bars = ax.bar(methods, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor='black')
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Inference Time Comparison (100 samples)', fontsize=16)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f'{m:.2f}s\n±{s:.2f}s', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'time_comparison.png', dpi=150)
    plt.close()

    # 柱状图：KL
    fig, ax = plt.subplots(figsize=(8, 6))
    means_kl = [statistics.mean(kl_fm), statistics.mean(kl_gram)]
    stds_kl = [statistics.stdev(kl_fm) if len(kl_fm) > 1 else 0,
               statistics.stdev(kl_gram) if len(kl_gram) > 1 else 0]
    bars = ax.bar(methods, means_kl, yerr=stds_kl, capsize=6, color=colors, alpha=0.85, edgecolor='black')
    ax.set_ylabel('KL Divergence (vs Content)', fontsize=14)
    ax.set_title('Color Preservation KL (100 samples, lower=better)', fontsize=16)
    for bar, m, s in zip(bars, means_kl, stds_kl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
                f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'kl_comparison.png', dpi=150)
    plt.close()

    print(f"\nAll figures saved to {out_dir.resolve()}")
    print("Done.")


if __name__ == '__main__':
    main()