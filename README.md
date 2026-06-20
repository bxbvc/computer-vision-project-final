# 风格迁移课程项目

## 简介

本项目是计算机视觉课程作业，围绕**神经风格迁移（Neural Style Transfer）**展开。风格迁移旨在将艺术作品的视觉风格迁移至图像，同时保持其语义结构完整。自从基于卷积神经网络的风格迁移方法开创以来，该领域经历了从全局统计量对齐方法、基于神经网络的方法，到最近基于扩散模型的方法。然而，现有方法本质上依然依赖于全局统计量对齐或风格损失监督，导致最终生成图像的颜色分布相较于原图发生偏移，且扩散模型的训练成本高昂，限制了实际应用。

针对上述问题，本项目首先复现和分析了 Gatys 的方法并对其进行改进，发现 2×10⁴ 的关键风格涌现点以及 ResNet-18 同样可以用于内容监督和风格监督。随后注意到自注意力公式和 Gram Matrix 的特殊关联，提出了新的模型架构 **CNNFormer**，其中自然涌现出内容-风格解耦特性。基于这一特征，设计了基于 **SSTS**、**CCDP** 和流匹配的风格迁移方案，并在 WikiArt 上进行了风格迁移实验。

---

## 项目结构

```
computer-vision/
├── checkpoints/              # 训练保存的模型权重
│   ├── taesd/                # TAESD 相关权重
│   ├── ct.pth                # 颜色纹理迁移模型 CCDP 权重
│   ├── decoder.pth           # SSTS Decoder 权重
│   ├── encoder.pth           # SSTS Encoder 权重
│   ├── fm_gram.pth           # 流匹配 + Gram Matrix 模型权重
│   ├── latest.pth            # 最近一次保存的模型
│   ├── resnet18.pth          # ResNet-18 权重
│   └── vgg.pth               # VGG 权重
├── cnnformer/                # CNNFormer 模型相关代码
│   └── train_cnnattn.py      # CNNFormer（CNN + Attention）训练脚本
├── color_texture_transfer/   # 颜色-纹理迁移方法
│   ├── batch_ct_transfer.py  # 批量颜色纹理迁移
│   ├── build_lut.py          # 构建颜色查找表（LUT）
│   ├── compare_ct_params.py  # 对比不同参数设置
│   ├── ct_benchmark.py       # 颜色纹理迁移基准测试
│   ├── ct_compare_lambda.py  # 对比不同 lambda 的影响
│   ├── ct_stats_class_kl.py  # 统计类别 KL 散度
│   ├── lut_transfer_single.py# 单张图像 LUT 迁移
│   ├── test_ct_transfer.py   # 颜色纹理迁移测试
│   ├── train_lut.py          # LUT 模型训练
│   └── viz_color_transfer.py # 可视化颜色迁移结果
├── flow_matching_transfer/   # 基于流匹配的风格迁移
│   ├── fm_benchmark.py       # 流匹配方法基准测试
│   ├── test_fm.py            # 流匹配推理测试
│   └── train_fm.py           # 流匹配模型训练
├── gatys/                    # Gatys 风格迁移复现与改进
│   ├── compare_vgg_resnet.py # 对比 VGG 与 ResNet 特征
│   ├── lbfgs.py              # L-BFGS 优化实现
│   ├── vgg_params.py         # VGG 超参数实验
│   └── viz_step.py           # 可视化优化过程
├── single_transfer/          # 基于自编码器的单图风格迁移
│   ├── batch_eval_texture_spilt.py  # 批量评估纹理分割
│   ├── ed_benchmark.py       # Encoder-Decoder 基准测试
│   ├── ed_transfer_single.py # 单张图像编码解码迁移
│   ├── train_decoder.py      # Decoder 训练
│   └── train_encoder.py      # Encoder 训练
├── data/                     # 数据集目录
│   ├── cifar-100-python/     # CIFAR-100 数据集
│   ├── coco/                 # COCO 数据集
│   ├── wikiart_images/       # WikiArt 风格图像数据集
│   └── dataset_dict.json     # 数据集索引文件
├── compare/                  # 对比实验输出目录
├── README.md                 # 本文件
├── compare_adam_lbfgs_time.py      # 对比 Adam 与 L-BFGS 优化时间
├── compare_kl.py             # 对比 KL 散度相关指标
├── compare_loss.py           # 对比不同损失函数
├── compare_speed.py          # 对比不同方法速度
├── compare_time.py           # 对比方法耗时
├── eval_encoder.py           # Encoder 评估
├── eval_lut_lambdas.py       # 评估 LUT 不同 lambda
└── fm_vs_gram.py             # 流匹配与 Gram Matrix 方法对比
```

---

## 各模块说明

- **gatys/**：复现 Gatys 等人基于优化的风格迁移方法，尝试使用 ResNet-18 替代 VGG 作为特征提取器，并可视化优化中间过程。
- **cnnformer/**：提出并训练 CNNFormer 模型，通过 CNN 与自注意力机制的结合，实现内容与风格的自然解耦。
- **color_texture_transfer/**：基于颜色-纹理解耦的风格迁移方法，包括 LUT 构建、训练和批量测试。
- **flow_matching_transfer/**：基于流匹配（Flow Matching）的生成式风格迁移方案。
- **single_transfer/**：基于 Encoder-Decoder 架构的快速单图风格迁移，训练独立的编码器和解码器。
- **根目录对比脚本**：以 `compare_*.py` 和 `eval_*.py` 命名的脚本用于横向对比不同方法的速度、损失、优化器效率等指标。

---

## 数据集

- 内容图像：COCO
- 风格图像：WikiArt
- 辅助分类/评估：CIFAR-100

---

## 运行环境

Python 3.x + PyTorch，具体依赖请参考各脚本中的 `import` 部分。训练前请确保数据集路径正确，并将预训练权重或训练好的模型放置于 `checkpoints/` 目录下。

---

## 模型权重说明

由于模型权重文件体积较大，本仓库未将 `checkpoints/` 目录中的权重上传至 Git。如需使用预训练权重，请联系作者：

- **学号**：23307130295

---

## 备注

本项目为课程作业实验代码，各子目录对应论文中的不同方法模块，便于复现和对比分析。
