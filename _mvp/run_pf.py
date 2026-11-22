# -*- coding: utf-8 -*-
"""MVP-M2~M5：死推算基线 + 粒子滤波 2×2（轮廓比对/DINOv2 打分 × 运动提示/随机走）。

判据（预先定死）：
  1) 带提示 PF 末端误差中位数 ≤10mm 且比死推算低 ≥50%
  2) 带提示比随机走中位误差低 ≥30%（先验价值）
  3) DINOv2 组峰值显存 <6GB、每帧 ≤3s
  4) 不达标如实报
产出：results.md、fig1_误差曲线.png、fig2_先验价值.png
"""
import os
import re
import sys
import json
import time
import numpy as np
import pybullet as p
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
N = 150
INIT_SPREAD = np.deg2rad(3.0)
SIG_B_MAX = np.deg2rad(1.5)   # 带提示版：偏差抖动初值（退火起点，探索）
SIG_B_MIN = np.deg2rad(0.05)  # 退火终点（精雕，决定最终分辨率）
SIG_I = np.deg2rad(1.5)    # 随机走版：扩散步长（覆盖真实运动速度 ~0.42°/步的 3 倍余量，公平基线）


# ---------- 基础工具 ----------
def resize_nn(img, size):
    idx = (np.linspace(0, img.shape[0] - 1, size)).astype(int)
    return img[np.ix_(idx, idx)]


def rgb_to_hs_hist(img, bins=(16, 8)):
    """numpy 手写 HSV 直方图（不引 cv2）。返回归一化 (16,8)。"""
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    df = mx - mn + 1e-9
    h = np.where(mx == r, ((g - b) / df) % 6,
                 np.where(mx == g, (b - r) / df + 2, (r - g) / df + 4)) * 60.0
    s = np.where(mx > 1e-6, df / (mx + 1e-9), 0.0)
    hh = np.clip((h / 360.0 * bins[0]).astype(int), 0, bins[0] - 1)
    ss = np.clip((s * bins[1]).astype(int), 0, bins[1] - 1)
    hist = np.zeros(bins, np.float64)
    np.add.at(hist, (hh.ravel(), ss.ravel()), 1.0)
    return hist / (hist.sum() + 1e-9)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


class FastScorer:
    """快版打分：整图像素比对（零域差下最强的"画出来比对"；探针实测真值处差异=0.0）。
    演化记录：HS 直方图太钝（权重恒匀）→ 轮廓 IoU 边缘（2°偏差只差2-3像素压不住噪声）→ 像素比对。"""
    size = 128
    name = "像素比对"

    def prep_obs(self, frames):
        return [resize_nn(f, self.size).astype(np.float32) for f in frames]

    def score(self, obs_feat, imgs):
        x = np.stack([resize_nn(im, self.size).astype(np.float32) for im in imgs])
        return -np.abs(x - obs_feat[None]).mean((1, 2, 3))


class DinoScorer:
    size = 224
    name = "DINOv2"

    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()

    @torch.no_grad()
    def embed(self, imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().cuda() / 255.0
        x = (x - MEAN) / STD
        f = self.model(x)
        return torch.nn.functional.normalize(f, dim=1)

    @torch.no_grad()
    def prep_obs(self, frames):
        return self.embed(np.stack([resize_nn(f, self.size) for f in frames]))

    @torch.no_grad()
    def score(self, obs_feat, imgs):
        f = self.embed(imgs)
        return (f * obs_feat).sum(1).float().cpu().numpy()


# ---------- 渲染 + 正运动学（一次循环拿齐）----------
def render_ee(poses, size, body, joint_ids, ee_link, view, proj):
    imgs = np.empty((len(poses), size, size, 3), np.uint8)
    ees = np.empty((len(poses), 3))
    for i, q in enumerate(poses):
        for k, j in enumerate(joint_ids):
            p.resetJointState(body, j, float(q[k]))
        _, _, rgb, _, _ = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER)
        imgs[i] = np.reshape(rgb, (size, size, 4))[:, :, :3].astype(np.uint8)
        ees[i] = np.array(p.getLinkState(body, ee_link)[4])
    return imgs, ees


def systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), pos)


def softmax_w(s):
    # 温度=0.5×std：纯 τ=std 自废武功（打分越锐权重越平）；0.25 又过早坍缩（Neff→11 锁死错粒子）
    tau = max(0.5 * float(np.std(s)), 1e-6)
    e = np.exp((s - s.max()) / tau)
    return e / e.sum()


def fk_single(q, body, joint_ids, ee_link):
    for k, j in enumerate(joint_ids):
        p.resetJointState(body, j, float(q[k]))
    return np.array(p.getLinkState(body, ee_link)[4])


def run_pf(tag, scorer, use_prior, data, body, joint_ids, ee_link, view, proj, rng):
    q_true, ee_true = data["q_true"], data["ee_true"]
    q_enc0, q_enc_traj = data["q_enc0"], data["q_enc_traj"]
    T = len(q_true)
    obs_feats = scorer.prep_obs(data["frames"])
    errs = np.empty(T)
    t0_all = time.time()
    if scorer.name == "DINOv2":
        torch.cuda.reset_peak_memory_stats()
    if use_prior:
        # 带提示：状态=6 维恒定偏差 b（运动提示把整条轨迹变成已知，未知只剩标定误差）。
        # 静态状态演化=退火进化策略：每帧一代"选择→变异"，权重逐代作废，证据靠代际选择累积。
        state = rng.normal(0, INIT_SPREAD, (N, 6))
    else:
        # 随机走（DiffPF 式"动作不可得"）：状态=6 维动态关节角，盲目扩散
        state = q_enc0[None, :] + rng.normal(0, INIT_SPREAD, (N, 6))
    for t in range(T):
        t0 = time.time()
        if use_prior:
            particles = q_enc_traj[t][None, :] - state
        else:
            particles = state = state + rng.normal(0, SIG_I, (N, 6))
        imgs, ees = render_ee(particles, scorer.size, body, joint_ids, ee_link, view, proj)
        s = scorer.score(obs_feats[t], imgs)
        w = softmax_w(s)
        ee_est = (w[:, None] * ees).sum(0)
        errs[t] = np.linalg.norm(ee_est - ee_true[t]) * 1000.0
        neff = 1.0 / np.sum(w ** 2)
        if use_prior or neff < N / 2:
            idx = systematic_resample(w, rng)
            state = state[idx]
            if use_prior:
                # 变异步只在选择后（退火：前期大步探索，后期精雕定分辨率）
                sig = SIG_B_MIN + (SIG_B_MAX - SIG_B_MIN) * np.exp(-t / 45.0)
                state = state + rng.normal(0, sig, (N, 6))
        if t % 20 == 0:
            print(f"[{tag}] frame {t}/{T}  err={errs[t]:.1f}mm  Neff={neff:.0f}  "
                  f"{(time.time()-t0)*1000:.0f}ms/帧", flush=True)
    vram = (torch.cuda.max_memory_allocated() / 1e9) if scorer.name == "DINOv2" else 0.0
    return dict(errs=errs, ms_per_frame=(time.time() - t0_all) / T * 1000.0, vram_gb=vram)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK 控制台防炸（U+2212 等）
    z = np.load(os.path.join(HERE, "data.npz"))
    frames, q_true, ee_true = z["frames"], z["q_true"], z["ee_true"]
    meta = json.loads(str(z["meta"]))
    joint_ids, ee_link = meta["joint_ids"], meta["ee_link"]
    T = len(q_true)

    # 预埋偏差：跟踪侧起始读数 = 真值 + 每关节 ±2° 恒定偏差（电机读数不准）
    bias = np.deg2rad(meta["bias_deg"]) * np.array(meta["bias_signs"], float)
    q_enc0 = q_true[0] + bias
    du = np.zeros_like(q_true)
    du[1:] = np.diff(q_true, axis=0)   # 关节增量读数（完美增量=免费的运动提示）
    q_enc_traj = q_enc0[None, :] + np.cumsum(du, axis=0)  # 逐帧电机读数=真值+恒定偏差
    data = dict(frames=frames, q_true=q_true, ee_true=ee_true, du=du, q_enc0=q_enc0, q_enc_traj=q_enc_traj)

    urdf = os.path.join(HERE, "ur5_patched.urdf")
    assert os.path.exists(urdf), "缺 ur5_patched.urdf，请先跑 gen_data.py"
    p.connect(p.DIRECT)
    # 场景必须与 gen_data 完全一致（白底+红臂），否则粒子渲染的是另一个世界，打分器必然失明
    white = [1.0, 1.0, 1.0, 1.0]
    for pos, eul, half in [((0.3, 0.3, -0.06), (0, 0, 0), [3, 3, 0.05]),
                           ((-1.2, 0.3, 1.2), (0, 1.5708, 0), [3, 3, 0.05]),
                           ((0.3, 1.8, 1.2), (1.5708, 0, 0), [3, 3, 0.05])]:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=white)
        p.createMultiBody(0, 0, vis, pos, p.getQuaternionFromEuler(eul))
    body = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    red = [0.85, 0.10, 0.10, 1.0]
    for j in range(p.getNumJoints(body)):
        p.changeVisualShape(body, j, rgbaColor=red)
    p.changeVisualShape(body, -1, rgbaColor=red)
    view = p.computeViewMatrix([1.15, -1.15, 0.95], [0.35, 0.0, 0.35], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, 1.0, 0.05, 4.0)

    # ---- M2 死推算基线：起始就被偏差带偏，只积分增量，不看图 ----
    print("[M2] 死推算基线 ...", flush=True)
    q_dr = q_enc0[None, :] + np.cumsum(du, axis=0)
    errs_dr = np.empty(T)
    for t in range(T):
        ee = fk_single(q_dr[t], body, joint_ids, ee_link)
        errs_dr[t] = np.linalg.norm(ee - ee_true[t]) * 1000.0
    print(f"[M2] 完成 中位误差={np.median(errs_dr):.1f}mm", flush=True)

    results = {"死推算（不看图）": dict(errs=errs_dr, ms_per_frame=0.0, vram_gb=0.0)}

    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        # 烟测：前 100 帧、只跑 像素-带提示，快速验证架构收敛与退火地板
        data = {k: (v[:100] if k in ("frames", "q_true", "ee_true", "du", "q_enc_traj") else v)
                for k, v in data.items()}
        r = run_pf("SMOKE-像素-带提示", FastScorer(), True, data, body, joint_ids, ee_link,
                   view, proj, np.random.default_rng(20260924))
        print(f"SMOKE median_mm={np.median(r['errs']):.1f} last_mm={r['errs'][-1]:.1f}", flush=True)
        return

    # ---- M3/M4 粒子滤波 2×2 ----
    hist, dino = FastScorer(), None
    for tag, scorer, use_prior in [
        ("PF-像素-带提示", hist, True),
        ("PF-像素-随机走", hist, False),
        ("PF-DINOv2-带提示", None, True),
        ("PF-DINOv2-随机走", None, False),
    ]:
        if scorer is None:
            if dino is None:
                print("[M3] 加载冻结 DINOv2-S ...", flush=True)
                dino = DinoScorer()
            scorer = dino
        print(f"[M3/M4] {tag} 开跑", flush=True)
        results[tag] = run_pf(tag, scorer, use_prior, data, body, joint_ids, ee_link,
                             view, proj, np.random.default_rng(20260924))
        print(f"[M3/M4] {tag} 完成 中位误差={np.median(results[tag]['errs']):.1f}mm", flush=True)

    p.disconnect()

    # ---- M5 出表出图 + 判据裁定 ----
    lines = ["# MVP 结果（自动汇总，数字照实填）", "",
             f"- 设置：N={N}，初始散布 ±3°，预埋偏差 ±2°/关节；带提示版=偏差状态化+退火抖动 2°→0.05°（q_t=读数_t−b），"
             f"随机走版=动态关节角扩散 {np.rad2deg(SIG_I):.1f}°/步；300 帧正弦轨迹，观测=渲染图（零域差）", ""]
    lines.append("| 方法 | 中位误差(mm) | 均值(mm) | 95分位(mm) | 每帧耗时(ms) | 峰值显存(GB) |")
    lines.append("|---|---|---|---|---|---|")
    order = ["死推算（不看图）", "PF-像素-带提示", "PF-像素-随机走", "PF-DINOv2-带提示", "PF-DINOv2-随机走"]
    for k in order:
        e, r = results[k]["errs"], results[k]
        lines.append(f"| {k} | {np.median(e):.1f} | {np.mean(e):.1f} | {np.percentile(e,95):.1f} "
                     f"| {r['ms_per_frame']:.0f} | {r['vram_gb']:.2f} |")

    med = {k: float(np.median(results[k]["errs"])) for k in order}
    dr, pfh, idh, pfd, idd = (med["死推算（不看图）"], med["PF-像素-带提示"], med["PF-像素-随机走"],
                              med["PF-DINOv2-带提示"], med["PF-DINOv2-随机走"])
    c1a = pfd <= 10.0 and pfd <= 0.5 * dr
    c1b = pfh <= 10.0 and pfh <= 0.5 * dr
    c2a = pfh <= 0.7 * idh
    c2b = pfd <= 0.7 * idd
    dino_run = results["PF-DINOv2-带提示"]
    c3 = dino_run["vram_gb"] < 6.0 and dino_run["ms_per_frame"] <= 3000.0
    lines += ["", "## 判据裁定（预先定死，不赛后改标准）", "",
              f"1. 主判据（≤10mm 且比死推算低≥50%）：DINOv2 组 {pfd:.1f}mm vs 算基线 {dr:.1f}mm → **{'通过' if c1a else '不通过'}**；"
              f"像素组 {pfh:.1f}mm → **{'通过' if c1b else '不通过'}**",
              f"2. 先验价值（带提示比随机走低≥30%）：像素 {pfh:.1f} vs {idh:.1f} → **{'通过' if c2a else '不通过'}**；"
              f"DINOv2 {pfd:.1f} vs {idd:.1f} → **{'通过' if c2b else '不通过'}**",
              f"3. 笔记本可行（显存<6GB、每帧≤3s）：{dino_run['vram_gb']:.2f}GB、{dino_run['ms_per_frame']:.0f}ms/帧 → **{'通过' if c3 else '不通过'}**",
              "",
              "注：观测=自渲染（仿真内零域差），本表只证机制通不通，不背真实世界承诺。"]
    open(os.path.join(HERE, "results.md"), "w", encoding="utf-8").write("\n".join(lines))
    np.savez(os.path.join(HERE, "curves.npz"),
             **{"m%d" % i: results[k]["errs"] for i, k in enumerate(order)})
    print("\n".join(lines), flush=True)

    # 图1：五条误差曲线
    plt.figure(figsize=(9, 5))
    for k in order:
        plt.plot(results[k]["errs"], label=k, linewidth=1.2)
    plt.xlabel("帧序"); plt.ylabel("末端位置误差 (mm)")
    plt.title("末端跟踪误差随时间变化（300 帧）"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig1_误差曲线.png"), dpi=140)

    # 图2：先验价值（同打分下 带提示 vs 随机走）
    plt.figure(figsize=(9, 5))
    for k, style in [("PF-像素-带提示", "-"), ("PF-像素-随机走", "--"),
                     ("PF-DINOv2-带提示", "-"), ("PF-DINOv2-随机走", "--")]:
        plt.plot(results[k]["errs"], style, label=k, linewidth=1.2)
    plt.xlabel("帧序"); plt.ylabel("末端位置误差 (mm)")
    plt.title("运动提示的价值：同打分下 带提示(实线) vs 随机走(虚线)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig2_先验价值.png"), dpi=140)
    print("[M5] 图表已出：fig1_误差曲线.png / fig2_先验价值.png / results.md", flush=True)


if __name__ == "__main__":
    main()
