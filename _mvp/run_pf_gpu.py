# -*- coding: utf-8 -*-
"""run_pf_gpu.py — GPU 渲染版五组实验（真显卡光栅化画粒子）。

与 run_pf.py（CPU/TinyRenderer 版）逐位同源：同一 data.npz 的 q_true/ee_true/预埋偏差、
同一随机种子 20260924、同一退火/重采样/温度参数——唯一变量=渲染器（Tiny 28ms/帧 → GPU 0.5ms/帧）。
观测与粒子渲染都走 GpuRenderer（同渲染器零域差）；白盒背景以白色清屏等价（两边一致）。
白盒三件套不再需要（画面里只贡献纯白像素=清屏色）。

判据（与 CPU 版同口径，预先定死）：
  1) 带提示 PF 末端误差中位数 ≤10mm 且比死推算低 ≥50%
  2) 带提示比随机走中位误差低 ≥30%（先验价值）
  3) 峰值显存 <6GB、每帧 ≤3s（实时性）
  4) 不达标如实报
产出：results_gpu.md、fig1_gpu_误差曲线.png、fig2_gpu_先验价值.png、curves_gpu.npz
"""
import os
import sys
import json
import time
import numpy as np
import pybullet as p
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpu_render import GpuRenderer

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
N = 150
INIT_SPREAD = np.deg2rad(3.0)
SIG_B_MAX = np.deg2rad(1.5)   # 带提示版：偏差抖动初值（退火起点）
SIG_B_MIN = np.deg2rad(0.05)  # 退火终点（精雕分辨率）
SIG_I = np.deg2rad(1.5)       # 随机走版：扩散步长（与 CPU 版逐位同参）


def resize_nn(img, size):
    idx = (np.linspace(0, img.shape[0] - 1, size)).astype(int)
    return img[np.ix_(idx, idx)]


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


class FastScorer:
    """快版打分：整图像素比对（同渲染器零域差下最强的画出来比对）。"""
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


def fk_single(q, body, joint_ids, ee_link):
    for k, j in enumerate(joint_ids):
        p.resetJointState(body, j, float(q[k]))
    return np.array(p.getLinkState(body, ee_link)[4])


def render_ee_gpu(gr, poses, size, body, joint_ids, ee_link, view, proj):
    """GPU 渲染一批姿态的图 + 各姿态末端真值位置（正运动学由 pybullet 提供）。"""
    imgs = np.empty((len(poses), size, size, 3), np.uint8)
    ees = np.empty((len(poses), 3))
    for i, q in enumerate(poses):
        imgs[i] = gr.render(q, size, view, proj)
        ees[i] = fk_single(q, body, joint_ids, ee_link)
    return imgs, ees


def systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), pos)


def softmax_w(s):
    # 温度=0.5×std（与 CPU 版同参；τ=std 自废武功）
    tau = max(0.5 * float(np.std(s)), 1e-6)
    e = np.exp((s - s.max()) / tau)
    return e / e.sum()


def run_pf(tag, scorer, use_prior, data, gr, body, joint_ids, ee_link, view, proj, rng):
    q_true, ee_true = data["q_true"], data["ee_true"]
    q_enc0, q_enc_traj = data["q_enc0"], data["q_enc_traj"]
    T = len(q_true)
    obs_feats = scorer.prep_obs(data["frames"])
    errs = np.empty(T)
    t0_all = time.time()
    if scorer.name == "DINOv2":
        torch.cuda.reset_peak_memory_stats()
    if use_prior:
        state = rng.normal(0, INIT_SPREAD, (N, 6))
    else:
        state = q_enc0[None, :] + rng.normal(0, INIT_SPREAD, (N, 6))
    for t in range(T):
        t0 = time.time()
        if use_prior:
            particles = q_enc_traj[t][None, :] - state
        else:
            particles = state = state + rng.normal(0, SIG_I, (N, 6))
        imgs, ees = render_ee_gpu(gr, particles, scorer.size, body, joint_ids, ee_link, view, proj)
        s = scorer.score(obs_feats[t], imgs)
        w = softmax_w(s)
        ee_est = (w[:, None] * ees).sum(0)
        errs[t] = np.linalg.norm(ee_est - ee_true[t]) * 1000.0
        neff = 1.0 / np.sum(w ** 2)
        if use_prior or neff < N / 2:
            idx = systematic_resample(w, rng)
            state = state[idx]
            if use_prior:
                sig = SIG_B_MIN + (SIG_B_MAX - SIG_B_MIN) * np.exp(-t / 45.0)
                state = state + rng.normal(0, sig, (N, 6))
        if t % 20 == 0:
            print(f"[{tag}] frame {t}/{T}  err={errs[t]:.1f}mm  Neff={neff:.0f}  "
                  f"{(time.time()-t0)*1000:.0f}ms/帧", flush=True)
    vram = (torch.cuda.max_memory_allocated() / 1e9) if scorer.name == "DINOv2" else 0.0
    return dict(errs=errs, ms_per_frame=(time.time() - t0_all) / T * 1000.0, vram_gb=vram)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(os.path.join(HERE, "data.npz"))
    frames_ignored, q_true, ee_true = z["frames"], z["q_true"], z["ee_true"]  # frames 不用：观测改走 GPU 渲染
    meta = json.loads(str(z["meta"]))
    joint_ids, ee_link = meta["joint_ids"], meta["ee_link"]
    T = len(q_true)

    # 预埋偏差与 CPU 版逐位同源（同一 data.npz 的 meta）
    bias = np.deg2rad(meta["bias_deg"]) * np.array(meta["bias_signs"], float)
    q_enc0 = q_true[0] + bias
    du = np.zeros_like(q_true)
    du[1:] = np.diff(q_true, axis=0)
    q_enc_traj = q_enc0[None, :] + np.cumsum(du, axis=0)

    urdf = os.path.join(HERE, "ur5_patched.urdf")
    assert os.path.exists(urdf), "缺 ur5_patched.urdf，请先跑 gen_data.py"
    p.connect(p.DIRECT)
    body = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)  # 仅用其正运动学；颜色/背景由 GPU 渲染器负责
    view = p.computeViewMatrix([1.15, -1.15, 0.95], [0.35, 0.0, 0.35], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, 1.0, 0.05, 4.0)
    gr = GpuRenderer(body, urdf, joint_ids)
    print(f"[GPU] GL_RENDERER = {gr.info['GL_RENDERER']}", flush=True)

    # ---- M2 死推算基线（纯运动学，不渲染）----
    print("[M2] 死推算基线 ...", flush=True)
    q_dr = q_enc0[None, :] + np.cumsum(du, axis=0)
    errs_dr = np.empty(T)
    for t in range(T):
        ee = fk_single(q_dr[t], body, joint_ids, ee_link)
        errs_dr[t] = np.linalg.norm(ee - ee_true[t]) * 1000.0
    print(f"[M2] 完成 中位误差={np.median(errs_dr):.1f}mm", flush=True)
    results = {"死推算（不看图）": dict(errs=errs_dr, ms_per_frame=0.0, vram_gb=0.0)}

    # ---- M3/M4 五组（观测按各打分器工作分辨率现渲，同渲染器零域差）----
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
        print(f"[M3/M4] {tag} 开跑（观测 {T} 帧 GPU 现渲 @ {scorer.size}²）", flush=True)
        obs = render_ee_gpu(gr, q_true, scorer.size, body, joint_ids, ee_link, view, proj)[0]
        data = dict(frames=obs, q_true=q_true, ee_true=ee_true,
                    q_enc0=q_enc0, q_enc_traj=q_enc_traj)
        results[tag] = run_pf(tag, scorer, use_prior, data, gr, body, joint_ids, ee_link,
                             view, proj, np.random.default_rng(20260924))
        print(f"[M3/M4] {tag} 完成 中位误差={np.median(results[tag]['errs']):.1f}mm", flush=True)

    p.disconnect()

    # ---- M5 出表出图 + 判据裁定（同 CPU 版口径）----
    lines = ["# GPU 渲染版结果（自动汇总，数字照实填）", "",
             "- 与 CPU 版（run_pf.py）逐位同源：同一 data.npz 轨迹/偏差、种子 20260924、同退火参数；"
             "唯一变量=渲染器（TinyRenderer→真 GPU 光栅，观测与粒子同渲染器零域差）", ""]
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
    lines += ["", "## 判据裁定（预先定死，与 CPU 版同口径）", "",
              f"1. 主判据（≤10mm 且比死推算低≥50%）：DINOv2 组 {pfd:.1f}mm vs 基线 {dr:.1f}mm → **{'通过' if c1a else '不通过'}**；"
              f"像素组 {pfh:.1f}mm → **{'通过' if c1b else '不通过'}**",
              f"2. 先验价值（带提示比随机走低≥30%）：像素 {pfh:.1f} vs {idh:.1f} → **{'通过' if c2a else '不通过'}**；"
              f"DINOv2 {pfd:.1f} vs {idd:.1f} → **{'通过' if c2b else '不通过'}**",
              f"3. 笔记本可行（显存<6GB、每帧≤3s）：{dino_run['vram_gb']:.2f}GB、{dino_run['ms_per_frame']:.0f}ms/帧 → **{'通过' if c3 else '不通过'}**",
              "",
              "注：观测=GPU 现渲（同渲染器零域差），本表只证机制，不背真实相机承诺（那是 Apollo 抽查的事）。"]
    open(os.path.join(HERE, "results_gpu.md"), "w", encoding="utf-8").write("\n".join(lines))
    np.savez(os.path.join(HERE, "curves_gpu.npz"),
             **{"m%d" % i: results[k]["errs"] for i, k in enumerate(order)})
    print("\n".join(lines), flush=True)

    plt.figure(figsize=(9, 5))
    for k in order:
        plt.plot(results[k]["errs"], label=k, linewidth=1.2)
    plt.xlabel("帧序"); plt.ylabel("末端位置误差 (mm)")
    plt.title("GPU 渲染版：末端跟踪误差（300 帧）"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig1_gpu_误差曲线.png"), dpi=140)

    plt.figure(figsize=(9, 5))
    for k, style in [("PF-像素-带提示", "-"), ("PF-像素-随机走", "--"),
                     ("PF-DINOv2-带提示", "-"), ("PF-DINOv2-随机走", "--")]:
        plt.plot(results[k]["errs"], style, label=k, linewidth=1.2)
    plt.xlabel("帧序"); plt.ylabel("末端位置误差 (mm)")
    plt.title("GPU 渲染版：运动提示的价值（带提示实线 vs 随机走虚线）")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig2_gpu_先验价值.png"), dpi=140)
    print("[M5] 图表已出", flush=True)


if __name__ == "__main__":
    main()
