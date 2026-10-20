# Ultrasound 3DGS 复现结果（UltraGS / UltraG-Ray）

在 **RTX 3060 Laptop 6GB** 上复现两篇超声 3D Gaussian Splatting 工作的训练日志、指标与对比渲染。

| 项目 | 论文 | 上游代码 |
|---|---|---|
| **UltraGS** | *UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis* (ICME 2026) | https://github.com/Bean-Young/UltraGS |
| **UltraG-Ray** | *UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis* (MIDL 2026) | https://github.com/jakobkla/UltraG-Ray |

本仓库**只放结果**（指标 / 日志 / 渲染图），不含数据集与权重。

---

## 环境

| 项 | 值 |
|---|---|
| 系统 | Windows 11 + WSL2 (Ubuntu 22.04) |
| GPU | RTX 3060 Laptop，6 GB（sm_86） |
| UltraGS | conda env `ultrags`，Python 3.10，torch 2.0.0 + CUDA 11.8 |
| UltraG-Ray | conda env `ultragray`，Python 3.10，torch 2.4.1 + CUDA 12.1 |
| 数据 | UltraGS：`us-in-the-wild`（10 个膝扫序列，HuggingFace）<br>UltraG-Ray：`pig_shoulder_v2` / `spine_phantom`（官方 Google Drive） |

环境搭建的全部踩坑记录见 [`ultrags/复现记录.md`](ultrags/复现记录.md)（7 个坑：torch 装成 CPU 版、MKL 符号缺失、CCCL 头文件缺失、simple_knn 漏提交 `__init__.py`、libnvrtc 软链等）。

---

## 结果一：UltraGS（`us-in-the-wild`，10 个 sweep）

训练配置：30,000 iterations，`data/wild/sweep_XX` → `output/sweep_XX`。

### sweep_08（已完成）

| 指标 | 值 |
|---|---|
| PSNR | **27.383** |
| SSIM | **0.8520** |
| LPIPS | **0.2818** |
| 训练时长 | 23 分钟 |
| 显存占用 | ~2.4–2.6 GB / 6 GB |

产物：[`ultrags/sweep_08/results.json`](ultrags/sweep_08/results.json)、[`per_view.json`](ultrags/sweep_08/per_view.json)、[`renders/`](ultrags/sweep_08/renders)、[`comparison.png`](ultrags/sweep_08/comparison.png)

### 其余 9 个 sweep

训练队列自动接力（ComReg 训练结束、GPU 空闲后启动），每个 sweep 约 **17–18 分钟**。

| sweep | 帧数 | PSNR | SSIM | LPIPS | 状态 |
|---|---|---|---|---|---|
| sweep_01 | 786 | **20.122** | 0.7315 | 0.4531 | ✅ |
| sweep_02 | 86 | **25.812** | 0.8403 | 0.3048 | ✅ |
| sweep_03 | 186 | **20.782** | 0.7470 | 0.3623 | ✅ |
| sweep_04 | 251 | — | — | — | 🔄 训练中 |
| sweep_05 | 435 | — | — | — | ⏳ 排队 |
| sweep_06 | 478 | — | — | — | ⏳ 排队 |
| sweep_07 | 98 | — | — | — | ⏳ 排队 |
| sweep_09 | 856 | — | — | — | ⏳ 排队 |
| sweep_10 | 309 | — | — | — | ⏳ 排队 |

> 观察：帧数多的 sweep（01/09 有 786/856 帧）PSNR 明显更低（20 左右），帧数少的（02/08 有 86/82 帧）能到 25–27。
> 这与 3DGS 的训练预算固定为 30k iterations 有关——帧越多，每次迭代见过的视角覆盖越稀疏。

- `sweep_01` 上次在 **iteration 3000** 处中断（半成品保留在本地 `output/sweep_01_partial_3000`）
- 原因：UltraGS 基于 3DGS，**只保存点云不保存优化器状态**，无法断点续训，必须从头重跑

> 渲染图每个 sweep 只保留 12 张均匀采样（完整渲染在本地 `output/sweep_XX/test/`），避免仓库体积失控。

---

## 结果二：UltraG-Ray

### pig_shoulder_v2（上次跑到 ~9,999 步，未完成）

验证集 104 个视角：

| step | num_GS | PSNR | SSIM |
|---|---|---|---|
| 999 | 500,000 | 23.578 | 0.7061 |
| 9,999 | 285,033 | **25.395** | **0.7680** |

产物：[`ultragray/pig_shoulder_v2/`](ultragray/pig_shoulder_v2)（`stats/*.json`、`stats/val_*.csv` 含 40+ 指标逐 step 记录、`comparison/` 渲染对比）

> 注意：`trainer.py` 的 `--ckpt` 参数**只做评估**（`runner.eval()`），不续训；最终指标需从头训练到 30,000 步。

### spine_phantom（未开始）

数据（`images_train.npy` / `poses_train.npy` / `images_val_v2.npy` / `poses_val_v2.npy`）已就位，尚未训练。

---

## 复现命令

### UltraGS

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultrags
cd ~/UltraGS

python train.py   -s data/wild/sweep_XX -m output/sweep_XX
python render.py  -s data/wild/sweep_XX -m output/sweep_XX --skip_train
python metrics.py -m output/sweep_XX
```

### UltraG-Ray

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultragray
cd ~/UltraG-Ray

python trainer.py --config_file data/pig_shoulder_v2/conf.json --disable_viewer
python trainer.py --config_file data/spine_phantom/conf.json   --disable_viewer
```

> `--disable_viewer` 必须加，否则训练结束后会 hang 在 viewer 上。

---

## 仓库结构

```
ultrags/
  sweep_08/            # 已完成 case 的指标 + 渲染
  复现记录.md           # 环境搭建与 7 个踩坑的完整记录
ultragray/
  pig_shoulder_v2/     # 部分训练（~9999 步）的指标与对比图
logs/                  # 训练队列日志（补齐后更新）
```

---

## 引用

```bibtex
@inproceedings{ultrags2026,
  title  = {UltraGS: Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis},
  booktitle = {ICME},
  year   = {2026}
}

@inproceedings{ultragray2026,
  title  = {UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis},
  booktitle = {MIDL},
  year   = {2026}
}
```
