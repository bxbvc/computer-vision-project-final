import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
from torchvision.utils import save_image
from PIL import Image
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import linalg
from diffusers import AutoencoderTiny

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 可配置参数 ====================
WIDTH_FACTOR = 2
DEPTH_FACTOR = 2
NUM_SAMPLES = 100          # 评估图片数
SAMPLE_STEPS = 20          # 流匹配采样步数
USE_AMP = True

# ==================== 工具函数 ====================

def make_divisible(v, divisor=32):
    return max(divisor, int(v + divisor // 2) // divisor * divisor)


# ==================== 模型定义（与训练代码一致） ====================

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = nn.GroupNorm(min(32, out_ch), out_ch)

    def forward(self, x):
        return self.gn(self.pointwise(self.depthwise(x)))


class TimeEmbed(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.SiLU(),
            nn.Linear(512, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, half, device=device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class CTBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_cond=False, cond_c=192):
        super().__init__()
        self.use_cond = use_cond
        qv_in = cond_c if use_cond else in_c

        self.q1 = DSConv(qv_in, qv_in, 3, 1, 1)
        self.q2 = DSConv(qv_in, out_c, 3, stride, 1)

        self.v1 = DSConv(qv_in, qv_in, 3, 1, 1)
        self.v2 = DSConv(qv_in, out_c, 3, stride, 1)

        if stride != 1 or in_c != out_c:
            self.res = DSConv(in_c, out_c, 3, stride, 1)
        else:
            self.res = None
        self.out_gn = nn.GroupNorm(min(32, out_c), out_c)

    def forward(self, x, t_emb, cond_s=None, cond_c=None):
        if self.use_cond:
            q = self.q2(F.silu(self.q1(cond_s + t_emb)))
            v = self.v2(F.silu(self.v1(cond_c + t_emb)))
        else:
            q = self.q2(F.silu(self.q1(x + t_emb)))
            v = self.v2(F.silu(self.v1(x + t_emb)))

        B, Co, Hp, Wp = q.shape
        N = Hp * Wp
        qf = q.reshape(B, Co, N)
        vf = v.reshape(B, Co, N)

        attn = torch.bmm(qf.transpose(1, 2), qf) / (Co ** 0.5)
        attn = F.softmax(attn + 1e-6, dim=-1)

        out = torch.bmm(vf, attn.transpose(1, 2)).reshape(B, Co, Hp, Wp)

        res = self.res(x) if self.res is not None else x
        return F.silu(self.out_gn(out + res))


class CTDecBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.compress = DSConv(in_c, out_c, 1, 1, 0)

        self.q1 = DSConv(out_c, out_c, 3, 1, 1)
        self.q2 = DSConv(out_c, out_c, 3, 1, 1)

        self.v1 = DSConv(out_c, out_c, 3, 1, 1)
        self.v2 = DSConv(out_c, out_c, 3, 1, 1)

        self.out_gn = nn.GroupNorm(min(32, out_c), out_c)

    def forward(self, x, t_emb):
        x = self.compress(x)

        q = self.q2(F.silu(self.q1(x + t_emb)))
        v = self.v2(F.silu(self.v1(x + t_emb)))

        B, C, H, W = q.shape
        N = H * W
        qf = q.reshape(B, C, N)
        vf = v.reshape(B, C, N)

        attn = torch.bmm(qf.transpose(1, 2), qf) / (C ** 0.5)
        attn = F.softmax(attn + 1e-6, dim=-1)

        out = torch.bmm(vf, attn.transpose(1, 2)).reshape(B, C, H, W)
        return F.silu(self.out_gn(out + x))


class ContentEncoder(nn.Module):
    def __init__(self, base_ch=64, out_ch=192):
        super().__init__()
        self.c1 = DSConv(3, base_ch, 3, 2, 1)
        self.c2 = DSConv(base_ch, out_ch, 3, 2, 1)
        self.c3 = DSConv(out_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = self.c3(x)
        return x


class ResNetFeat(nn.Module):
    """ResNet18 取 layer1 输出: 64 x H/4 x W/4"""
    def __init__(self):
        super().__init__()
        rn = models.resnet18(weights=None)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)
        return f1


class ScalableFlowMatchingNet(nn.Module):
    def __init__(self, width_factor=1.0, depth_factor=1.0):
        super().__init__()
        self.width_factor = width_factor
        self.depth_factor = depth_factor

        base_cfg = {'c0': 192, 'c1': 384, 'c2': 768, 'c3': 1536}
        self.c0 = make_divisible(base_cfg['c0'] * width_factor)
        self.c1 = make_divisible(base_cfg['c1'] * width_factor)
        self.c2 = make_divisible(base_cfg['c2'] * width_factor)
        self.c3 = make_divisible(base_cfg['c3'] * width_factor)

        self.n_enc0 = max(2, int(round(2 * depth_factor)))
        self.n_enc1 = max(2, int(round(2 * depth_factor)))
        self.n_enc2 = max(4, int(round(4 * depth_factor)))
        self.n_dec2 = max(3, int(round(3 * depth_factor)))
        self.n_dec1 = max(3, int(round(3 * depth_factor)))
        self.n_dec0 = max(3, int(round(3 * depth_factor)))

        self.time_embed = TimeEmbed(128)

        time_dims = []
        time_dims += [self.c0] + [self.c1] * (self.n_enc0 - 1)
        time_dims += [self.c1] + [self.c2] * (self.n_enc1 - 1)
        time_dims += [self.c2] + [self.c3] * (self.n_enc2 - 1)
        time_dims += [self.c2] * self.n_dec2
        time_dims += [self.c1] * self.n_dec1
        time_dims += [self.c0] * self.n_dec0

        self.time_projs = nn.ModuleList([nn.Linear(128, d) for d in time_dims])

        self.latent_aug = DSConv(4, self.c0, 1, 1, 0)
        self.pi_s = DSConv(64, self.c0, 3, 2, 1)
        base_ch = make_divisible(64 * width_factor)
        self.pi_c = ContentEncoder(base_ch=base_ch, out_ch=self.c0)

        self.enc0 = nn.ModuleList()
        self.enc0.append(CTBlock(self.c0, self.c1, stride=2, use_cond=True, cond_c=self.c0))
        for _ in range(1, self.n_enc0):
            self.enc0.append(CTBlock(self.c1, self.c1, stride=1, use_cond=False))

        self.enc1 = nn.ModuleList()
        self.enc1.append(CTBlock(self.c1, self.c2, stride=2, use_cond=False))
        for _ in range(1, self.n_enc1):
            self.enc1.append(CTBlock(self.c2, self.c2, stride=1, use_cond=False))

        self.enc2 = nn.ModuleList()
        self.enc2.append(CTBlock(self.c2, self.c3, stride=2, use_cond=False))
        for _ in range(1, self.n_enc2):
            self.enc2.append(CTBlock(self.c3, self.c3, stride=1, use_cond=False))

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.ModuleList()
        self.dec2.append(CTDecBlock(self.c3 + self.c2, self.c2))
        for _ in range(1, self.n_dec2):
            self.dec2.append(CTDecBlock(self.c2, self.c2))

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.ModuleList()
        self.dec1.append(CTDecBlock(self.c2 + self.c1, self.c1))
        for _ in range(1, self.n_dec1):
            self.dec1.append(CTDecBlock(self.c1, self.c1))

        self.up0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec0 = nn.ModuleList()
        self.dec0.append(CTDecBlock(self.c1 + self.c0, self.c0))
        for _ in range(1, self.n_dec0):
            self.dec0.append(CTDecBlock(self.c0, self.c0))

        self.out = DSConv(self.c0, 4, 3, 1, 1)

    def forward(self, z_raw, t, cond_s, cond_c):
        B = z_raw.shape[0]
        H16, W16 = z_raw.shape[2], z_raw.shape[3]
        H8, W8 = H16 // 2, W16 // 2
        H4, W4 = H8 // 2, W8 // 2
        H2, W2 = H4 // 2, W4 // 2

        t_vec = self.time_embed(t)
        t_embs = [proj(t_vec)[:, :, None, None] for proj in self.time_projs]

        t_idx = 0
        def get_t_emb(ch, h, w):
            nonlocal t_idx
            te = t_embs[t_idx].expand(B, ch, h, w)
            t_idx += 1
            return te

        z_aug = self.latent_aug(z_raw)
        s = self.pi_s(cond_s)

        skips = {}
        x = z_aug
        for i, blk in enumerate(self.enc0):
            h = H16 if i == 0 else H8
            w = W16 if i == 0 else W8
            c = self.c0 if i == 0 else self.c1
            t_emb = get_t_emb(c, h, w)
            if i == 0:
                x = blk(x, t_emb, s, cond_c)
                skips['enc0'] = x
            else:
                x = blk(x, t_emb)

        for i, blk in enumerate(self.enc1):
            h = H8 if i == 0 else H4
            w = W8 if i == 0 else W4
            c = self.c1 if i == 0 else self.c2
            t_emb = get_t_emb(c, h, w)
            x = blk(x, t_emb)
            if i == len(self.enc1) - 1:
                skips['enc1'] = x

        for i, blk in enumerate(self.enc2):
            h = H4 if i == 0 else H2
            w = W4 if i == 0 else W2
            c = self.c2 if i == 0 else self.c3
            t_emb = get_t_emb(c, h, w)
            x = blk(x, t_emb)

        x = self.dec2[0](torch.cat([self.up2(x), skips['enc1']], dim=1), get_t_emb(self.c2, H4, W4))
        for blk in self.dec2[1:]:
            x = blk(x, get_t_emb(self.c2, H4, W4))

        x = self.dec1[0](torch.cat([self.up1(x), skips['enc0']], dim=1), get_t_emb(self.c1, H8, W8))
        for blk in self.dec1[1:]:
            x = blk(x, get_t_emb(self.c1, H8, W8))

        x = self.dec0[0](torch.cat([self.up0(x), z_aug], dim=1), get_t_emb(self.c0, H16, W16))
        for blk in self.dec0[1:]:
            x = blk(x, get_t_emb(self.c0, H16, W16))

        return self.out(x)


# ==================== FID 计算工具 ====================

class InceptionV3Feature(nn.Module):
    """提取 InceptionV3 池化层特征 (2048 维) 用于 FID"""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        inception.eval()
        inception.fc = nn.Identity()  # 移除分类头，保留 2048 维特征
        self.model = inception
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 3, H, W], range [0, 1]
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feat = self.model(x)
        return feat  # [B, 2048]


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算 Fréchet Inception Distance"""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"[Warning] sqrtm 不收敛，添加正则化 eps={eps}")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def extract_features(images, extractor, batch_size=8, desc="Extracting"):
    """分批提取特征，带进度条"""
    feats = []
    n = len(images)
    # 获取 device，兼容不同 PyTorch 版本
    device = next(extractor.model.parameters()).device
    for i in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        batch = torch.stack(images[i:i+batch_size]).to(device)
        with torch.no_grad():
            f = extractor(batch).cpu()
        feats.append(f)
    return torch.cat(feats, dim=0).numpy()


# ==================== 主函数 ====================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    coco_root = r'data\coco\train2017'
    ckpt_dir = Path('checkpoints')
    out_dir = Path('fid_eval')
    out_dir.mkdir(exist_ok=True)

    # ---- 1. 随机选 100 张图 ----
    cp = Path(coco_root)
    all_paths = [p for p in cp.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')]
    if len(all_paths) < NUM_SAMPLES:
        raise ValueError(f"COCO 目录只有 {len(all_paths)} 张图，不足 {NUM_SAMPLES} 张")
    selected_paths = random.sample(all_paths, NUM_SAMPLES)
    print(f"从 COCO 随机选取了 {NUM_SAMPLES} 张图用于评估")

    # ---- 2. 预处理 ----
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---- 3. 加载 ResNet18 ----
    resnet_path = ckpt_dir / 'resnet18.pth'
    if not resnet_path.exists():
        raise FileNotFoundError(f"找不到 ResNet18 权重: {resnet_path}")
    rn = models.resnet18(weights=None)
    rn.load_state_dict(torch.load(resnet_path, map_location='cpu'))
    resnet = ResNetFeat().to(device)
    resnet.load_state_dict(rn.state_dict(), strict=False)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    print("ResNet18 加载完成")

    # ---- 4. 加载 TAESD ----
    taesd_dir = ckpt_dir / 'taesd'
    if not (taesd_dir / 'config.json').exists():
        raise FileNotFoundError(f"找不到 TAESD: {taesd_dir}")
    taesd = AutoencoderTiny.from_pretrained(str(taesd_dir), local_files_only=True).to(device)
    taesd.eval()
    for p in taesd.parameters():
        p.requires_grad = False
    print("TAESD 加载完成")

    # ---- 5. 加载 Flow Matching 模型 ----
    fm_ckpt_path = ckpt_dir / 'fm_350_10_60.pth'
    if not fm_ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {fm_ckpt_path}")
    model = ScalableFlowMatchingNet(width_factor=WIDTH_FACTOR, depth_factor=DEPTH_FACTOR).to(device)
    ckpt = torch.load(fm_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flow Matching 模型加载完成 ({n_params/1e6:.1f}M 参数)")

    # ---- 6. 加载 InceptionV3 ----
    inception = InceptionV3Feature().to(device)
    print("InceptionV3 特征提取器加载完成")

    # ---- 7. 准备图片 ----
    real_images = []   # [0, 1] 范围
    fake_images = []   # [0, 1] 范围
    save_indices = random.sample(range(NUM_SAMPLES), min(5, NUM_SAMPLES))  # 随机保存 5 张对比

    print(f"\n开始生成 {NUM_SAMPLES} 张图片并计算 FID...")
    pbar = tqdm(enumerate(selected_paths), total=NUM_SAMPLES, desc="Generating", unit="img")

    for idx, path in pbar:
        img = Image.open(path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)  # [1, 3, 128, 128], [-1, 1]

        with torch.no_grad():
            x_01 = (x + 1.0) / 2.0  # 转到 [0, 1]
            real_images.append(x_01.squeeze(0).cpu())

            # 编码条件
            z1 = taesd.encode(x_01).latents
            z0 = torch.randn_like(z1)
            cond_s = resnet(x_01)
            cond_c = model.pi_c(x_01)

            # 流匹配采样（欧拉法）
            z = z0
            dt = 1.0 / SAMPLE_STEPS
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    for i in range(SAMPLE_STEPS):
                        ti = torch.full((1,), i * dt, device=device)
                        v = model(z, ti, cond_s, cond_c)
                        z = z + dt * v
            else:
                for i in range(SAMPLE_STEPS):
                    ti = torch.full((1,), i * dt, device=device)
                    v = model(z, ti, cond_s, cond_c)
                    z = z + dt * v

            recon = taesd.decode(z).sample  # [1, 3, 128, 128], [0, 1]
            fake_images.append(recon.squeeze(0).cpu())

            # 保存对比图
            if idx in save_indices:
                comp = torch.cat([x_01, recon], dim=0)
                save_image(comp, out_dir / f'compare_{idx:03d}.png', nrow=2, normalize=True, value_range=(0, 1))

        pbar.set_postfix(step=f"{idx+1}/{NUM_SAMPLES}")

    # ---- 8. 提取 Inception 特征 ----
    print("\n提取真实图片特征...")
    real_feats = extract_features(real_images, inception, batch_size=8, desc="Real")

    print("提取生成图片特征...")
    fake_feats = extract_features(fake_images, inception, batch_size=8, desc="Fake")

    # ---- 9. 计算 FID ----
    mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    print(f"\n{'='*40}")
    print(f"  评估样本数: {NUM_SAMPLES}")
    print(f"  采样步数:   {SAMPLE_STEPS}")
    print(f"  FID 分数:   {fid:.4f}")
    print(f"{'='*40}")
    print(f"\n对比图已保存到 {out_dir}/ 目录")
    print("提示: 100 张样本计算的 FID 仅供参考，标准评估通常使用 5000+ 张图以获得稳定估计。")


if __name__ == '__main__':
    main()