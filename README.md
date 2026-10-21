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

### 全部 sweep 结果（30,000 iterations，单卡 3060）

| sweep | 帧数 | PSNR | SSIM | LPIPS | 高斯点数 | 状态 |
|---|---|---|---|---|---|---|
| sweep_07 | 98 | **29.349** | 0.8461 | 0.2500 | ~36 万 | ✅ |
| sweep_08 | 82 | **27.383** | **0.8520** | 0.2818 | — | ✅ |
| sweep_04 | 251 | 26.004 | 0.7913 | 0.3721 | — | ✅ |
| sweep_02 | 86 | 25.812 | 0.8403 | 0.3048 | — | ✅ |
| sweep_06 | 478 | 23.745 | 0.7376 | 0.4077 | ~16 万 | ✅ |
| sweep_05 | 435 | 22.142 | 0.7269 | 0.4331 | ~11 万 | ✅ |
| sweep_03 | 186 | 20.782 | 0.7470 | 0.3623 | — | ✅ |
| sweep_09 | 856 | 20.564 | 0.7499 | 0.4396 | ~10 万 | ✅ |
| sweep_01 | 786 | 20.122 | 0.7315 | 0.4531 | — | ✅ |
| sweep_10 | 309 | — | — | — | **129 万** | ⚠️ 未完成 |

**规律**：PSNR 与帧数强负相关（98 帧 → 29.3，856 帧 → 20.6）。训练预算固定为 30k iterations，
帧越多则每个视角被采样次数越少。这与论文结论一致。

**sweep_10 异常说明**：该 sweep 在 densification 阶段高斯点数暴涨到 **1,295,591**（其他 sweep 是 9–36 万），
导致单次迭代从 0.035 s 恶化到 **26.75 s**。跑到 29,100/30,000 时估算仍需 6.7 小时，故人工终止。
训练曲线当时 test PSNR 22.16 / SSIM 0.707。重跑需调高 `--densify_grad_threshold` 限制点数增长。

> 渲染图每个 sweep 保留最多 12 张均匀采样（完整渲染在本地 `output/sweep_XX/test/`），避免仓库体积失控。
> 时间成本：正常 sweep 约 17–18 分钟；sweep_10 因点数爆炸 4 小时仍未完成。


---

## 结果二：UltraG-Ray

两个场景均训练到 **20,000 步**（原计划 30,000，因单步耗时随高斯点数增长而显著上升，主动封顶）。

### pig_shoulder_v2（20k，本次完成）

验证集 104 个视角：

| step | num_GS | PSNR | SSIM |
|---|---|---|---|
| 999 | 500,000 | 23.577 | 0.7061 |
| 9,999 | 284,874 | 25.442 | 0.7676 |
| **19,999** | 492,165 | **25.553** | **0.7708** |

产物：[`ultragray/pig_shoulder_v2/`](ultragray/pig_shoulder_v2)（`stats/val_*.csv` 含 40+ 指标逐 step 记录、`renders/` 取最终步的 gt/ultrasound/echo/trans 各 2 帧）

> **10k → 20k 只涨 0.11 dB**，已明显收敛。单步耗时从启动时的约 5 it/s 一路衰减到 **1.6 it/s**
> （高斯点数先被裁剪到 28 万，再增长回 49 万），因此 20k 封顶是合理折中。

### spine_phantom（20k，本次完成）

| step | num_GS | PSNR | SSIM |
|---|---|---|---|
| 999 | 500,000 | 24.153 | 0.9050 |
| 9,999 | 62,518 | 24.845 | 0.9156 |
| **19,999** | 101,569 | **25.920** | **0.9279** |

产物：[`ultragray/spine_phantom/`](ultragray/spine_phantom)

> 与 pig_shoulder 相反，spine 在 10k→20k 仍涨 **+1.08 dB**，属于**尚未收敛**。
> 差异根源在点数：spine 稳定在 ~10 万高斯，pig 涨到 49 万——点数少则训练快（40+ it/s vs 1.6 it/s），
> 同样的 20k 步对 spine 而言"训练量"更少。**封顶步数应按场景收敛情况定，不能一刀切。**

### 上次未完成运行（归档）

上一轮 pig_shoulder 只跑到 ~9,999 步（PSNR 25.395），产物归档在
[`ultragray/pig_shoulder_prev_9999/`](ultragray/pig_shoulder_prev_9999)（含 4 张对比图与早停指标）。

> 注意：`trainer.py` 的 `--ckpt` 参数**只做评估**（`runner.eval()`），不续训；要拿到完整结果只能从头训练。

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
