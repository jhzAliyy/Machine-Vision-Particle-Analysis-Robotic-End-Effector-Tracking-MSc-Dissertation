# results —— 终版结果

本目录只放论文与交付包引用的结果；历史输出已于 2026-11-10 按要求删除，这里没有第二份副本。

| 文件 | 是什么 | 怎么来的 |
|---|---|---|
| `results_all34.json` | 全 34 包 × 双臂 = 68 次运行的 3D 误差表（七点中位 / 单侧钳尖 / 钳口中心，各含中位、P95、逐帧池化），逐包逐臂；顶层 `protocol` 记配置与训练划分 | `pipeline/track_online.py`，配置 `CKPT=C:/Users/Aliyy Gingko/Desktop/workspace/space9_class/uk_supervisor/dissertation/runtime/models/detector_14kp_20261110.pt NCH=14 NORM=imagenet OBS=lift CAL=det PK=peak SIG_O=0.0015 CONF_W=1 MF_MODE=causal`，`PACKS` 缺省跑全量 |
| `phase_metrics.json` | Option A 四段分析全量数字：68 臂 332 episodes，逐臂逐 episode 四段时长/抓取/路径/速度/占比，效率（GT 标签一致性、逐段误差、127fps）与 27 组敏感性；`method` 字段记全部阈值与依据 | `pipeline/phase_metrics.py`（2026-11-22），日志 `tracking/run_phase_metrics_20261122.log` |
| `results_all34.log` | 上述运行的完整控制台记录 | 同上，`2>&1 \| tee` |
| `overlay_000021_PSM1.mp4` | 演示视频 1：000021 PSM1（单极弯剪，训练未见过），1001 帧全量，1406×493@30fps，全帧中位 6.1 mm | `DET=1 SPLIT=1 SEGS="000021:PSM1,000027:PSM1" pipeline/render_demo.py` |
| `overlay_000027_PSM1.mp4` | 演示视频 2：000027 PSM1（持针器，训练未见过），同上，全帧中位 3.7 mm | 同上 |
| `overlay_*_i0000/i0500/i1000.png` | 两段视频第 0 / 500 / 1000 帧静帧 | 同上 |
| `overlay_summary.json` | 两段视频的逐帧误差数组、中位值、帧号 | 同上 |
| `err_curves_3packs.png` | 附录曲线：000020（最难一包）、000021、000027 的逐帧误差与 15 帧滑动中位 | `VIDEO=0 SEGS="000020:PSM1,000021:PSM1,000027:PSM1" OUT_OV=runtime/curves pipeline/render_demo.py` 导出逐帧数值，再由 `pipeline/plot_err_curves.py` 画图 |

口径说明：

- 误差 = 估计的器械末端点到人工标注真值点的三维距离，单位 mm。三个口径同时保存：`err3d_mm.median` 是每帧七点取中位（旧报告引的就是这个，5.00 mm）、`tip_median` 只算远端钳尖（5 号点 / PSM3 的 11 号点，5.80 mm）、`jaw_median` 是两个钳尖中点（4–5 / 11–12，4.30 mm）；`pooled` 是全帧摊平后的同一三口径，与逐次运行中位相差不超过 0.1 mm。视频右下角显示的是最近 15 帧的中位（画面小字已标注），单个异常帧不会直接跳到读数上。逐帧原值在 `overlay_summary.json` 的 `err_mm_per_frame`。
- 训练划分：基础 20 包加 000021–000024 四个额外包，共 24 包（48 次运行）；未见的是 000020 与 000025–000033（10 包 / 20 次）。JSON 里每包的 `split` 字段只是“<20 为 train”的粗略划分，不要拿它当训练划分用。
- 视频片段选的是两段训练未见的包（000021、000027），全表中位 6.1 / 3.7 mm。全表最差的一包（000020 PSM1，中位 8.5 mm，17% 的帧超过 13 mm）不作为展示片段，其逐帧曲线在 `err_curves_3packs.png` 中。
- 视频里的全帧中位（6.2 / 3.9 mm）与全量表（6.1 / 3.7 mm）相差 0.1–0.2 mm，原因是全表只统计左右目都能三角化的点，视频统计全部有效帧。
- 本目录内容对应 2026-11-10 20:21 的产出。
