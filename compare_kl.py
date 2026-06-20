# compare_kl_100.py
import os
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
import torchvision.transforms.functional as TVF
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SEED = 42
NUM_SAMPLES = 100
FM_BATCH_SIZE = 10
N_STEPS = 20
LAMBDA = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

CKPT_DIR = Path("checkpoints")
COCO_ROOT = r"data\coco\train2017"
WIKI_ROOT = r"data\wikiart_images"

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def fmt4(x):
    return f"{x:.4g}"

def list_images(root, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    root = Path(root)
    files = [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)

# ========================== Gatys NST ==========================
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
            s_loss = sum(F.mse_loss(self.gram_matrix(a), b) for a, b in zip(s_feats, style_targets))
            loss = content_weight * c_loss + style_weight * s_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                input_img.clamp_(-1, 1)
        return input_img.detach()

# ========================== Flow Matching Model ==========================
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
                v_prev = F.interpolate(v_prev, size=q_cur.shape[-2:], mode="nearest")
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

# ========================== Color Model & Tools ==========================
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

def compute_kl_between(img_a_01, img_b_01, bins=33):
    H_a = splat_3d(img_a_01, bins).squeeze(0) + 1e-8
    H_b = splat_3d(img_b_01, bins).squeeze(0) + 1e-8
    return (H_a * (H_a.log() - H_b.log())).sum().item()

# ========================== Texture & Encoder-Decoder ==========================
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
        z = self.stem(x)
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        return z

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

# ========================== Texture Tools ==========================
def guided_filter(guide, src, radius=2, eps=1e-2):
    mean_I = F.avg_pool2d(guide, 2*radius+1, stride=1, padding=radius)
    mean_p = F.avg_pool2d(src, 2*radius+1, stride=1, padding=radius)
    mean_Ip = F.avg_pool2d(guide * src, 2*radius+1, stride=1, padding=radius)
    mean_II = F.avg_pool2d(guide * guide, 2*radius+1, stride=1, padding=radius)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = F.avg_pool2d(a, 2*radius+1, stride=1, padding=radius)
    mean_b = F.avg_pool2d(b, 2*radius+1, stride=1, padding=radius)
    q = mean_a * guide + mean_b
    return q

def guided_filter_rgb(guide, src, radius=2, eps=1e-2):
    C = src.shape[1]
    outs = []
    for c in range(C):
        q = guided_filter(guide, src[:, c:c+1], radius, eps)
        outs.append(q)
    return torch.cat(outs, dim=1)

def extract_bbox(mask):
    ys, xs = torch.where(mask)
    if len(ys) == 0:
        return None
    return ys.min().item(), xs.min().item(), ys.max().item(), xs.max().item()

def tile_patch(patch, target_h, target_w):
    C, h, w = patch.shape
    if h == 0 or w == 0:
        return torch.zeros(C, target_h, target_w, device=patch.device, dtype=patch.dtype)
    repeat_h = (target_h + h - 1) // h
    repeat_w = (target_w + w - 1) // w
    tiled = patch.repeat(1, repeat_h, repeat_w)
    return tiled[:, :target_h, :target_w]

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

def transfer_texture_rgb(content, style, texture_model,
                         n_clusters=8, gf_radius=2, gf_eps=0.01,
                         use_feather=True, feather_sigma=1.5):
    device = content.device
    B, C, H, W = content.shape
    assert B == 1
    guide_c = content.mean(dim=1, keepdim=True)
    guide_s = style.mean(dim=1, keepdim=True)
    base_c = guided_filter_rgb(guide_c, content, radius=gf_radius, eps=gf_eps)
    base_s = guided_filter_rgb(guide_s, style, radius=gf_radius, eps=gf_eps)
    detail_s = style - base_s
    with torch.no_grad():
        z_c = texture_model(content)
        z_s = texture_model(style)
    labels_c = kmeans_cluster(z_c[0], K=n_clusters, max_iter=10, merge_thresh=0.95)
    labels_s = kmeans_cluster(z_s[0], K=n_clusters, max_iter=10, merge_thresh=0.95)
    K_c = labels_c.max().item() + 1
    K_s = labels_s.max().item() + 1
    centers_c = []
    centers_s = []
    for k in range(K_c):
        mask = labels_c == k
        centers_c.append(z_c[0][:, mask].mean(dim=1) if mask.any() else torch.zeros(18, device=device))
    for k in range(K_s):
        mask = labels_s == k
        centers_s.append(z_s[0][:, mask].mean(dim=1) if mask.any() else torch.zeros(18, device=device))
    centers_c = torch.stack(centers_c)
    centers_s = torch.stack(centers_s)
    c_norm = F.normalize(centers_c, p=2, dim=1)
    s_norm = F.normalize(centers_s, p=2, dim=1)
    sim = c_norm @ s_norm.T
    match = sim.argmax(dim=1)
    detail_out = torch.zeros_like(base_c)
    for k in range(K_c):
        m = match[k].item()
        mask_c = (labels_c == k)
        mask_s = (labels_s == m)
        if not mask_c.any() or not mask_s.any():
            continue
        y1s, x1s, y2s, x2s = extract_bbox(mask_s)
        patch_s = detail_s[0, :, y1s:y2s+1, x1s:x2s+1].clone()
        patch_s = patch_s - patch_s.mean(dim=(1, 2), keepdim=True)
        y1c, x1c, y2c, x2c = extract_bbox(mask_c)
        h_c = y2c - y1c + 1
        w_c = x2c - x1c + 1
        tiled = tile_patch(patch_s, h_c, w_c)
        sub_mask = mask_c[y1c:y2c+1, x1c:x2c+1].unsqueeze(0).float()
        if use_feather:
            pad = int(feather_sigma * 3) | 1
            sub_mask = TVF.gaussian_blur(sub_mask, kernel_size=max(pad, 3), sigma=feather_sigma)
        region = detail_out[0, :, y1c:y2c+1, x1c:x2c+1]
        region_new = torch.where(sub_mask.bool().expand_as(region), tiled, region)
        detail_out[0, :, y1c:y2c+1, x1c:x2c+1] = region_new
    result = base_c + detail_out
    return result.clamp(-1, 1), labels_c, labels_s

# ========================== Main ==========================
def main():
    print(f"Device: {DEVICE}")

    coco_imgs = list_images(COCO_ROOT)
    wiki_imgs = list_images(WIKI_ROOT)
    if len(coco_imgs) < NUM_SAMPLES or len(wiki_imgs) < NUM_SAMPLES:
        raise RuntimeError(f"Need {NUM_SAMPLES} images: coco={len(coco_imgs)}, wiki={len(wiki_imgs)}")

    random.shuffle(coco_imgs)
    random.shuffle(wiki_imgs)
    content_paths = coco_imgs[:NUM_SAMPLES]
    style_paths = wiki_imgs[:NUM_SAMPLES]

    # Load models
    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained(str(CKPT_DIR / "taesd"), local_files_only=True).to(DEVICE)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False

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

    fm_model = ScalableFlowMatchingNet(depth_factor=2.0).to(DEVICE)
    fm_ckpt = torch.load(CKPT_DIR / "fm_gram.pth", map_location=DEVICE)
    fm_model.load_state_dict(fm_ckpt["model"], strict=False)
    fm_model.eval()
    for p in fm_model.parameters():
        p.requires_grad = False

    color_model = StyleColorPredictor(bins=33).to(DEVICE)
    color_model.load_state_dict(torch.load(CKPT_DIR / "latest.pth", map_location=DEVICE)["model"])
    color_model.eval()
    for p in color_model.parameters():
        p.requires_grad = False

    encoder = TextureSegmentor(dim=18, window_size=4).to(DEVICE)
    ckp_e = torch.load(CKPT_DIR / "encoder.pth", map_location=DEVICE)
    encoder.load_state_dict(ckp_e["model_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = Decoder(bottleneck_ch=18, skip_ch=18, hid_ch=32, num_skips=3).to(DEVICE)
    ckp_d = torch.load(CKPT_DIR / "decoder.pth", map_location=DEVICE)
    decoder.load_state_dict(ckp_d["model_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    gatys = GatysNST(DEVICE)

    # CSV
    out_path = Path("compare_kl_100.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "content_name", "style_name", "flow_kl", "ct_kl", "direct_kl", "gatys_kl"])

    # Preload all tensors
    content_tensors = []
    style_tensors = []
    for i in range(NUM_SAMPLES):
        ct = TRANSFORM(Image.open(content_paths[i]).convert("RGB")).unsqueeze(0).to(DEVICE)
        st = TRANSFORM(Image.open(style_paths[i]).convert("RGB")).unsqueeze(0).to(DEVICE)
        content_tensors.append(ct)
        style_tensors.append(st)

    # 1) Flow Matching (batch)
    fm_kls = [0.0] * NUM_SAMPLES
    content_z, content_cs, content_cc = [], [], []
    style_cs, style_cc = [], []
    for i in range(NUM_SAMPLES):
        c01 = (content_tensors[i] + 1.0) / 2.0
        s01 = (style_tensors[i] + 1.0) / 2.0
        with torch.no_grad():
            cs_c, cc_c = resnet_fm(c01)
            z = taesd.encode(c01).latents
            cs_s, cc_s = resnet_fm(s01)
        content_z.append(z)
        content_cs.append(cs_c)
        content_cc.append(cc_c)
        style_cs.append(cs_s)
        style_cc.append(cc_s)

    for b_start in range(0, NUM_SAMPLES, FM_BATCH_SIZE):
        b_end = min(b_start + FM_BATCH_SIZE, NUM_SAMPLES)
        actual_bs = b_end - b_start
        z_batch = torch.cat([content_z[i] for i in range(b_start, b_end)], dim=0)
        cs_c_batch = torch.cat([content_cs[i] for i in range(b_start, b_end)], dim=0)
        cc_c_batch = torch.cat([content_cc[i] for i in range(b_start, b_end)], dim=0)
        cs_s_batch = torch.cat([style_cs[i] for i in range(b_start, b_end)], dim=0)

        with torch.no_grad():
            z_noise = forward_ode(fm_model, z_batch, cs_c_batch, cc_c_batch, n_steps=N_STEPS)
            cond_s_mixed = LAMBDA * cs_s_batch + (1.0 - LAMBDA) * cs_c_batch
            z_stylized = reverse_ode(fm_model, z_noise, cond_s_mixed, cc_c_batch, n_steps=N_STEPS)
            gen = taesd.decode(z_stylized).sample * 2.0 - 1.0

        for j in range(actual_bs):
            idx = b_start + j
            gen_01 = (gen[j:j+1] + 1.0) / 2.0
            content_01 = (content_tensors[idx] + 1.0) / 2.0
            fm_kls[idx] = compute_kl_between(gen_01, content_01)

    # 2) CT (Color once, then Texture)
    ct_kls = [0.0] * NUM_SAMPLES
    with torch.no_grad():
        style_cat = torch.cat(style_tensors, dim=0)
        probs, _ = color_model(style_cat)

    color_results = [None] * NUM_SAMPLES
    for i in tqdm(range(NUM_SAMPLES), desc="Color LUT", leave=False):
        color_results[i] = transfer_color_lut(content_tensors[i], probs[i], style_tensors[i],
                                                bins=33, n_iter=800, lr=0.001, tv_weight=10)

    for i in tqdm(range(NUM_SAMPLES), desc="CT", leave=False):
        I_ct, _, _ = transfer_texture_rgb(
            color_results[i], style_tensors[i], encoder,
            n_clusters=8, gf_radius=2, gf_eps=0.01,
            use_feather=True, feather_sigma=1.5
        )
        ct_01 = (I_ct + 1.0) / 2.0
        content_01 = (content_tensors[i] + 1.0) / 2.0
        ct_kls[i] = compute_kl_between(ct_01, content_01)

    # 3) Direct (Encoder-Decoder)
    direct_kls = [0.0] * NUM_SAMPLES
    for i in tqdm(range(NUM_SAMPLES), desc="Direct", leave=False):
        ct = content_tensors[i]
        st = style_tensors[i]
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

            q1_m = 1.0 * q1_r + 0.0 * q1_a
            q2_m = 1.0 * q2_r + 0.0 * q2_a
            q3_m = 1.0 * q3_r + 0.0 * q3_a

            z_attn, skips_attn = encoder.forward_with_skips_and_style(ct, [q1_m, q2_m, q3_m])
            rec = decoder(z_attn, skips_attn)

        rec_01 = (rec + 1.0) / 2.0
        content_01 = (content_tensors[i] + 1.0) / 2.0
        direct_kls[i] = compute_kl_between(rec_01, content_01)

    # 4) Gatys
    gatys_kls = [0.0] * NUM_SAMPLES
    for i in tqdm(range(NUM_SAMPLES), desc="Gatys", leave=False):
        I_gatys = gatys.run(content_tensors[i], style_tensors[i],
                            steps=500, content_weight=1.0, style_weight=1e7, lr=0.01)
        gatys_01 = (I_gatys + 1.0) / 2.0
        content_01 = (content_tensors[i] + 1.0) / 2.0
        gatys_kls[i] = compute_kl_between(gatys_01, content_01)

    # Write CSV
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(NUM_SAMPLES):
            writer.writerow([
                i,
                Path(content_paths[i]).name,
                Path(style_paths[i]).name,
                fmt4(fm_kls[i]), fmt4(ct_kls[i]),
                fmt4(direct_kls[i]), fmt4(gatys_kls[i]),
            ])

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary over {NUM_SAMPLES} random pairs")
    print(f"{'='*60}")
    print(f"  Flow:   mean={np.mean(fm_kls):.4g}")
    print(f"  CT:     mean={np.mean(ct_kls):.4g}")
    print(f"  Direct: mean={np.mean(direct_kls):.4g}")
    print(f"  Gatys:  mean={np.mean(gatys_kls):.4g}")
    print(f"{'='*60}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()