# -*- coding: utf-8 -*-
"""apollo_calib_sweep.py — 配准扭转错位的定点标定扫描（E4）。

背景（E1-E3 判读）：打分器无辜（零域差 3×100%），真实照片轴 z 判别倒挂（37.5%，扰动反而加分）
= 配准带系统性旋转错位。本扫描在"掌+前臂"点集上搜后配准修正角 (绕 z 扭转, 绕 y 俯仰)，
以轴 z/轴 y 的 20° 判别命中率为指标，10 帧标定 / 10 帧留出验证（防在测试集上自欺）。

预测（可证伪）：若倒挂确由系统性错位引起，命中率随修正角呈单峰，峰位=真实错位角；
涂上修正角后留出帧判别应回升（x≥80% 一档）。若扫描无峰或留出不回升 → 错位假设被证伪，回滚 H 树。
用法：py311 apollo_calib_sweep.py
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apollo_rotate_diag import depth_residual_score, project, rot_cands  # 复用同源打分与候选

NPZ = _os.path.join(_AW, "apollo_extract.npz")
HAND = _os.path.join(_AW, "hand_points.npy")


def rot_mat(axis, deg):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.eye(3)
    a, b = (axis + 1) % 3, (axis + 2) % 3
    R[a, a] = R[b, b] = c
    R[a, b], R[b, a] = -s, s
    return R


def hit_rate(pts, depths, gts, K, probe_axis):
    hits, tot = 0, 0
    for i in range(len(gts)):
        scores = [(m, depth_residual_score(*project(pts, T, K), depths[i])) for m, T in rot_cands(gts[i], probe_axis)]
        s0 = scores[0][1]
        for m, s in scores[1:]:
            if round(np.degrees(m), 1) == 20.0:
                tot += 1
                hits += (s0 > s)
    return hits / max(tot, 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    depth, gt, K = z["depth"], z["gt"], z["K"]
    hp = np.load(HAND)
    pf = json.loads(str(z["palm_frames"]))
    T_fix = np.linalg.inv(np.array(pf["/R_ARM_PALM|/R_PALM"]))
    gts = [g @ T_fix for g in gt]
    pts = hp   # 布局已是掌+前臂（指节排除）
    n = len(gts)
    cal, hold = slice(0, n // 2), slice(n // 2, n)
    print(f"掌+前臂 {len(pts)} 点 | 标定帧 {n//2} / 留出帧 {n - n//2}")

    grids = {2: np.arange(-5, 70.1, 5.0), 1: np.arange(-25, 25.1, 2.5)}   # z 首轮峰在 +25° 边界，延长；y 已判
    for axis, probe in ((2, 2), (1, 1)):   # 先扫 z（倒挂主轴），再扫 y
        grid = grids[axis]
        print(f"\n== 扫描绕{'xyz'[axis]}修正角（探{'xyz'[probe]}轴判别，标定帧） ==", flush=True)
        curve = []
        for deg in grid:
            R = rot_mat(axis, deg)
            p2 = (R @ pts.T).T
            curve.append(hit_rate(p2, depth[cal], [g for g in gts[cal]], K, probe))
        for d, h in zip(grid, curve):
            print(f"  {d:+5.1f}° → {h*100:5.1f}%")
        best = float(grid[int(np.argmax(curve))])
        peak = max(curve)
        base = float(curve[list(grid).index(0.0)])
        print(f"峰位={best:+.1f}° 峰值={peak*100:.1f}%（零修正={base*100:.1f}%）")
        # 留出验证
        R = rot_mat(axis, best)
        p2 = (R @ pts.T).T
        h_hold = hit_rate(p2, depth[hold], [g for g in gts[hold]], K, probe)
        h_hold0 = hit_rate(pts, depth[hold], [g for g in gts[hold]], K, probe)
        print(f"留出帧验证：涂修正 {h_hold*100:.1f}% vs 不涂 {h_hold0*100:.1f}% → "
              f"{'错位假设成立' if h_hold > h_hold0 + 0.05 else '修正无增益——错位假设在此轴被证伪'}")


if __name__ == "__main__":
    main()
