# -*- coding: utf-8 -*-
"""apollo_misalign_diag.py — 系统错配 D 的逐帧一致性诊断与求解。

pose_scan 的发现：score(GT)=-135mm（点云与深度错配 13.5cm），
手系平移 z 峰值偏 -3cm（增益 58.6mm）——H1 系统错配成立候选。
若"深度最优位姿 = GT @ D"在 20 帧上恒定，则 D 就是 3.28cm/23.46° 的主病根。

本脚本：
  ① 逐帧单轴细扫（平移 ±5cm 步 1cm；旋转 ±30° 步 5°，手系轴，一次一轴）
  ② 一致性：各帧峰值偏移的均值/std——std 小 = 系统性（可修），大 = 素材问题
  ③ 若系统性：迭代组合求 D（旋转→平移两轮），报告修正后 score 与判别裕度
产出 $APOLLO_WORK/apollo_misalign_diag.json
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys

import numpy as np

from apollo_hand_build2 import NPZ
from apollo_track import rot_about, rot_err_deg

OUT_JSON = _os.path.join(_AW, "apollo_misalign_diag.json")

TRANS_GRID = np.linspace(-0.05, 0.05, 11)      # ±5cm 步 1cm
ROT_GRID = np.arange(-30, 30.1, 5.0)            # ±30° 步 5°


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    K, gts, depths = z["K"], z["gt"], z["depth"]
    hp = np.load(_os.path.join(_AW, "hand_points.npy"))

    def score_at(T, depth, pts=None):
        p = hp if pts is None else pts
        pc = (T[:3, :3] @ p.T).T + T[:3, 3]
        okz = pc[:, 2] > 0.05
        uv = (K @ pc[okz].T).T
        u, v = np.round(uv[:, 0] / uv[:, 2]).astype(int), np.round(uv[:, 1] / uv[:, 2]).astype(int)
        h, w = depth.shape
        ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_obs = depth[v[ing], u[ing]]
        good = np.isfinite(d_obs) & (d_obs > 0.05)
        if good.sum() < 50:
            return -1.0
        return -float(np.mean(np.abs(pc[okz][ing][good, 2] - d_obs[good])))

    # ---- ① 逐帧单轴峰值偏移 ----
    n = len(gts)
    print("== ① 逐帧单轴细扫（手系轴，一次一轴；每帧每轴的 score 峰值偏移） ==", flush=True)
    peaks = {f"t{a}": [] for a in "xyz"}
    peaks.update({f"r{a}": [] for a in "xyz"})
    for i in range(n):
        T0, dep = gts[i], depths[i]
        for ai, ax in enumerate(np.eye(3)):
            cs = [score_at(_tr(T0, ax, dt), dep) for dt in TRANS_GRID]
            peaks[f"t{'xyz'[ai]}"].append(float(TRANS_GRID[int(np.argmax(cs))]))
            cs = [score_at(_ro(T0, ax, dth), dep) for dth in ROT_GRID]
            peaks[f"r{'xyz'[ai]}"].append(float(ROT_GRID[int(np.argmax(cs))]))
    print("轴   峰值偏移均值±std（平移 cm / 旋转 °）", flush=True)
    for k, v in peaks.items():
        v = np.array(v)
        unit = "cm" if k[0] == "t" else "°"
        scale = 100.0 if k[0] == "t" else 1.0
        print(f"  {k}: {v.mean()*scale:+7.2f} ± {v.std()*scale:5.2f} {unit}   逐帧={np.round(v*scale,1).tolist()}", flush=True)

    # ---- ② 一致性判定 ----
    consistent = all(np.array(v).std() < (0.02 if k[0] == "t" else 10.0) for k, v in peaks.items())
    print(f"\n一致性判定（平移 std<2cm 且旋转 std<10°）: {'系统性错配——可修' if consistent else '帧间不一致——素材/时差问题'}", flush=True)

    # ---- ③ 组合求解 D（迭代：先旋转后平移，两轮） ----
    print("\n== ③ 组合 D 迭代求解（每帧独立求 D_i，再看散布） ==", flush=True)
    rng = np.random.default_rng(20260924)
    Dis = []
    for i in range(n):
        T0, dep = gts[i], depths[i]
        D = np.eye(4)
        for _round in range(2):
            for ai, ax in enumerate(np.eye(3)):          # 旋转
                cs = [score_at(T0 @ D @ rot_about(ax, np.radians(dth)), dep) for dth in ROT_GRID]
                best = ROT_GRID[int(np.argmax(cs))]
                A = np.eye(4)
                D = D @ rot_about(ax, np.radians(best)) if best != 0.0 else D
            for ai, ax in enumerate(np.eye(3)):          # 平移
                cs = [score_at(T0 @ D @ _t4(ax, dt), dep) for dt in TRANS_GRID]
                best = TRANS_GRID[int(np.argmax(cs))]
                D = D @ _t4(ax, best) if best != 0.0 else D
        Dis.append(D)
    # 散布：各 D_i 相对平均的旋转/平移散布
    Rm = np.mean([D[:3, :3] for D in Dis], 0)
    U, _, Vt = np.linalg.svd(Rm)
    Ddet = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Ddet[2, 2] = -1
    Ravg = Vt.T @ Ddet @ U.T
    rots = [rot_err_deg(np.eye(4) @ D, _pad(Ravg)) for D in [d for d in Dis]]
    trs = np.array([D[:3, 3] for D in Dis])
    print(f"  D_i 平移散布(std): x={trs[:,0].std()*1000:.1f} y={trs[:,1].std()*1000:.1f} z={trs[:,2].std()*1000:.1f} mm", flush=True)
    print(f"  D_i 旋转散布(mean): {np.mean(rots):.2f}°", flush=True)
    Davg = np.eye(4)
    Davg[:3, :3] = Ravg
    Davg[:3, 3] = trs.mean(0)
    ang = rot_err_deg(np.eye(4), Davg)
    print(f"  D 平均: 平移=({Davg[0,3]*1000:.1f},{Davg[1,3]*1000:.1f},{Davg[2,3]*1000:.1f})mm  旋转={ang:.2f}°", flush=True)

    # ---- ④ 修正效果：score 与判别裕度 ----
    s_before = np.mean([score_at(gts[i], depths[i]) for i in range(n)])
    s_after = np.mean([score_at(gts[i] @ Davg, depths[i]) for i in range(n)])
    print(f"\nscore 修正前 {s_before*1000:.2f}mm → 修正后 {s_after*1000:.2f}mm（残差降低 {1000*(s_before-s_after):.2f}mm）", flush=True)

    def margin_eval(use_d):
        m = {a: [] for a in "xyz"}
        for i in range(n):
            T = gts[i] @ Davg if use_d else gts[i]
            s0 = score_at(T, depths[i])
            for ai, ax in enumerate(np.eye(3)):
                sp = [score_at(T @ rot_about(ax, s * np.radians(20.0)), depths[i]) for s in (+1, -1)]
                m["xyz"[ai]].append(s0 - max(sp))
        return {a: float(np.mean(m[a])) for a in "xyz"}

    m0, m1 = margin_eval(False), margin_eval(True)
    print(f"判别裕度(相机系±20°): 修正前 y={m0['y']*1000:+.2f} z={m0['z']*1000:+.2f} mm → 修正后 y={m1['y']*1000:+.2f} z={m1['z']*1000:+.2f} mm", flush=True)

    json.dump(dict(peaks={k: [float(x) for x in v] for k, v in peaks.items()},
                   consistent=bool(consistent),
                   D_avg=Davg.tolist(), D_rot_deg=float(ang),
                   D_spread_rot=float(np.mean(rots)), D_spread_trans_mm=trs.std(0).tolist(),
                   score_before_mm=float(s_before * 1000), score_after_mm=float(s_after * 1000),
                   margin_before=m0, margin_after=m1),
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存 {OUT_JSON}", flush=True)


def _tr(T, ax, dt):
    M = T.copy()
    M[:3, 3] = M[:3, 3] + M[:3, :3] @ (dt * ax)
    return M


def _ro(T, ax, dth):
    return T @ rot_about(ax, np.radians(dth))


def _t4(ax, dt):
    M = np.eye(4)
    M[:3, 3] = dt * ax
    return M


def _pad(R):
    M = np.eye(4)
    M[:3, :3] = R
    return M


if __name__ == "__main__":
    main()
