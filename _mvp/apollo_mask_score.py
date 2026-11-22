# -*- coding: utf-8 -*-
"""apollo_mask_score.py — 掩码观测项：给朝向装上眼睛。

诊断链结论：逐点深度差对掌块朝向弱可观测（扁平+自对称），且原均值被
背景离群污染。掩码/轮廓对朝向敏感：扁平掌块绕相机 y 轴转 20°，
投影轮廓长宽互换，掩码对称差剧变。

观测项定义（每帧）：
  M_obs  = |depth − z_hand| < 0.25m 的像素（z_hand=点云深度中位数）
  M_mod  = 点云投影光栅（每点半径 2px）
  mask_score = −对称差像素数 / sqrt(M_obs 像素数)   （尺度归一）

对照（20 帧，±20° 扰动，相机系轴）：
  S_depth 原均值口径 / S_depth 截尾 20% / S_mask / 组合(截尾+掩码)
判别裕度 = score(GT) − max score(扰动)。y/z 显著为正 = 朝向可观测。

产出 $APOLLO_WORK/apollo_mask_score.json
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys

import numpy as np

from apollo_hand_build2 import NPZ
from apollo_track import rot_about

OUT_JSON = _os.path.join(_AW, "apollo_mask_score.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    K, gts, depths = z["K"], z["gt"], z["depth"]
    hp = np.load(_os.path.join(_AW, "hand_points.npy"))

    def project(T):
        pc = (T[:3, :3] @ hp.T).T + T[:3, 3]
        okz = pc[:, 2] > 0.05
        uv = (K @ pc[okz].T).T
        u, v = np.round(uv[:, 0] / uv[:, 2]).astype(int), np.round(uv[:, 1] / uv[:, 2]).astype(int)
        return pc[okz], u, v

    def s_depth_mean(T, depth):
        pc, u, v = project(T)
        h, w = depth.shape
        ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_obs = depth[v[ing], u[ing]]
        good = np.isfinite(d_obs) & (d_obs > 0.05)
        if good.sum() < 50:
            return -1.0
        return -float(np.mean(np.abs(pc[ing][good, 2] - d_obs[good])))

    def s_depth_trim(T, depth, trim=0.2):
        pc, u, v = project(T)
        h, w = depth.shape
        ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_obs = depth[v[ing], u[ing]]
        good = np.isfinite(d_obs) & (d_obs > 0.05)
        if good.sum() < 50:
            return -1.0
        r = np.sort(np.abs(pc[ing][good, 2] - d_obs[good]))
        return -float(r[:int(len(r) * (1 - trim))].mean())

    def s_mask(T, depth):
        pc, u, v = project(T)
        h, w = depth.shape
        z_hand = float(np.median(pc[:, 2]))
        m_obs = np.isfinite(depth) & (np.abs(depth - z_hand) < 0.25)
        m_mod = np.zeros((h, w), bool)
        ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        for x, y in zip(u[ing], v[ing]):
            y0, y1 = max(0, y - 2), min(h, y + 3)
            x0, x1 = max(0, x - 2), min(w, x + 3)
            m_mod[y0:y1, x0:x1] = True
        n_obs = int(m_obs.sum())
        if n_obs < 200:
            return -1.0
        sym = int((m_obs ^ m_mod).sum())
        return -float(sym) / np.sqrt(n_obs)

    scores = {"S_depth原均值": s_depth_mean, "S_depth截尾20%": s_depth_trim, "S_掩码": s_mask}

    def combo(T, depth):
        return s_depth_trim(T, depth) + 0.004 * s_mask(T, depth)

    scores["S_组合(截尾+0.004*掩码)"] = combo

    print("== 各观测项判别裕度(±20°，相机系轴，20 帧均值) ==", flush=True)
    table = {}
    for name, fn in scores.items():
        m = {a: [] for a in "xyz"}
        for i in range(len(gts)):
            s0 = fn(gts[i], depths[i])
            for ai, ax in enumerate(np.eye(3)):
                sp = [fn(gts[i] @ rot_about(ax, s * np.radians(20.0)), depths[i]) for s in (+1, -1)]
                m["xyz"[ai]].append(s0 - max(sp))
        row = {a: float(np.mean(m[a])) for a in "xyz"}
        table[name] = row
        print(f"  {name}: x={row['x']:9.3f}  y={row['y']:9.3f}  z={row['z']:9.3f}", flush=True)

    json.dump(table, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
