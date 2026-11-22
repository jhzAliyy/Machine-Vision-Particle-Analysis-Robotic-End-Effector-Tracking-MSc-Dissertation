# 随机种子序列存档（MVP）

日期：2026-09-24 ｜ 范围：`_mvp/` 全部随机源（gen_data.py / run_pf.py / check_pixel.py / gpu_probe.py，检索式 `seed|rng|random` 已穷尽）

## 一、种子表

| 环节 | 位置 | 种子/随机源 | 用途 |
|---|---|---|---|
| 数据生成 gen_data.py | 全文件 | **无随机源** | 轨迹 q0/幅/频/相位全写死（gen_data.py:96-99）；预埋偏差 bias_deg=2.0、符号 [1,-1,1,-1,1,-1] 写死（gen_data.py:114-115）。TinyRenderer 渲染确定性（check_pixel.py 实测真值帧重渲差异 = 0.0） |
| PF 五组 run_pf.py | run_pf.py:222（smoke）、run_pf.py:241（全量 ×4） | `np.random.default_rng(20260924)` | 每组各自新建**同值**生成器。配对设计：组间差异来自算法，不来自运气 |
| 零域差探针 check_pixel.py | 全文件 | 无随机源 | 重渲真值帧做像素比对 |
| GPU 探针 gpu_probe.py | 全文件 | 无随机源 | 计时姿态 = q_true[::15] 确定性抽 20 个 |

## 二、序列与消耗顺序（run_pf.py，种子 20260924，N=150）

**带提示组**（偏差状态化：q_t = 读数_t − b，静态 6 维）：
1. 初始化：`rng.normal(0, INIT_SPREAD, (N,6))` 一次（INIT_SPREAD=3°）——run_pf.py:142
2. 逐帧：`systematic_resample` 消耗 `rng.random()` 标量 1 次（run_pf.py:113→159）→ 变异 `rng.normal(0, sig, (N,6))`（run_pf.py:164，sig 退火 1.5°→0.05°，τ_decay=45 帧）
3. 权重逐代作废（选择→变异进化式，无跨帧累积，无弗兰肯斯坦权重）

**随机走组**（动态 6 维关节角）：
1. 初始化：`rng.normal(0, INIT_SPREAD, (N,6))` 一次，锚在 q_enc0——run_pf.py:145
2. 逐帧：扩散 `rng.normal(0, SIG_I, (N,6))`（run_pf.py:151，SIG_I=1.5°）→ 若 Neff<N/2 则 `systematic_resample` 标量 1 次

**无其它随机源**：DINOv2 冻结 eval 前向、TinyRenderer、正运动学均确定性；torch 未设 manual_seed（无随机层，不需要）。

## 三、基准产物指纹（SHA256，2026-09-24 取，复核锚点）

| 文件 | SHA256 |
|---|---|
| data.npz | 1ca379fb327459d9bfc9ccac205a3a95c48afaeb216d4a6579748d568c345a09 |
| curves.npz | 91e65b149bde4ddb3f6c8e2d3babbbba9f02d0ced5cc5ff862546e1ad3079f4a |
| results.md | 826976d1c017c4d4135afce33b4b0cb05b544cfd3490f05717da07567be6395a |
| run_pf.py | 5947e33991e9b413b05d309d37508b39296d3f973e305fb1d079fc07254968fa |
| gen_data.py | 63c712d3f8884a23ec1626f1502fe69a82e877fbd3483bd3c45c24bbf3639e3e |

## 四、复核记录

1. **前一会话（全量两次）**：五组误差数组逐位一致（同种子重跑实测）。
2. **2026-09-24 smoke 复核**（像素-带提示 前 100 帧，`run_pf.py smoke`，零写盘）：
   - 锚点 = 全量跑同帧打印值：t=0/20/40/60/80 → err=14.0/9.5/16.9/16.8/24.9 mm，Neff=52/52/44/62/60；M2 死推算中位 24.1mm
   - 实测：**逐点一致（通过）**——err 14.0/9.5/16.9/16.8/24.9、Neff 52/52/44/62/60、M2 24.1mm 全部对上，打印精度下零偏差
   - 新锚点入档：SMOKE median_mm=17.9、last_mm=23.8（100 帧中位/末帧，打印一位小数）
3. DINOv2 组的逐位一致性证据 = 前一会话全量两次对比；本轮 smoke 只覆盖像素组（快路径），GPU/CUDA 前向不涉随机。

## 五、复现指令

```
# 本目录路径已全英文，直接运行即可（不需要盘符映射）
py311 run_pf.py smoke        # 种子复核（约 7 分钟，零写盘）
py311 run_pf.py              # 全量五组（约 2 小时，会重写 results.md / curves.npz / 两张图）
py311 gen_data.py            # 重新生成 data.npz（无随机源，理论上逐位一致；本轮未做二次生成比对以免覆写基准）
```

注：smoke 模式在 run_pf.py:217-223 提前 return，不触碰 results.md/curves.npz/图，可随时跑。

## 六、2026-09-24 精细手+全量测试阶段的随机源

| 环节 | 位置 | 种子 | 用途 |
|---|---|---|---|
| 掌/前臂表面采样 | apollo_hand_build.py 与 apollo_hand_build2.py 内 `default_rng(20260924)` | 20260924 | 前臂胶囊圆周采样相位（网格采样由 trimesh 自带随机，已显式播种） |
| 全量跟踪退火进化 | apollo_track.py `default_rng(20260924)` | 20260924 | 150 粒子初始化、逐帧选择（searchsorted 重采样 u0）、变异抖动 |
| 旋转诊断 E1--E3 | apollo_rotate_diag.py | 无随机 | 确定性扰动网格（±5/10/20° × 3 轴） |
| 标定扫描 E4 | apollo_calib_sweep.py | 无随机 | 确定性角度网格（±25°/−5~70°），标定/留出各 10 帧固定对半分 |
| 素材抽取/标定 | apollo_extract.py / apollo_probe.py / bag_*.py | 无随机 | 时间戳最近邻同步、双观测标定，全确定性 |

复现链：`py311 apollo_extract.py <real bag>` → `py311 apollo_hand_build2.py` → `py311 apollo_track.py`；
诊断链：`py311 apollo_rotate_diag.py`、`py311 apollo_calib_sweep.py`。
产物锚点（已入 `apollo_out/`）：hand_points.npy（1600 点精细手，含 +25° 钟差常量）、apollo_track_result.json（全量测试判分）。
