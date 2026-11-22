# -*- coding: utf-8 -*-
"""apollo_pose_scan.py — GT 邻域位姿扫描：定位朝向不可观测的病根。

动机：V0 判别裕度 y=-16mm / z=-12mm——GT 位姿下点云平均深度残差反而比
±20° 扰动位姿大 16mm。三种病根假设，扫描可一步分清：
  H1 系统错配：score 峰值明显偏离 GT(平移偏移>1cm 或旋转偏移>5°)
     → 病在标定 T_bh2cam / T_hm2bh 或点云尺寸，修源头，修好后三轴自然可观测
  H2 朝向不可观测：某轴 score 曲线平坦(±10° 内变化 < 2mm)
     → 病在观测模型，必须加观测项(掩码/轮廓)
  H3 滤波器问题：峰值在 GT 且尖锐
     → 病在跟踪链，改重采样/退火

口径：20 帧(apollo_extract.npz)，单轴变其余冻结，score=负平均深度残差。
产出 $APOLLO_WORK/apollo_pose_scan.json
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys

import numpy as np

from apollo_hand_build2 import NPZ
from apollo_track import rot_about

OUT_JSON = _os.path.join(_AW, "apollo_pose_scan.json")

# 扫描网格：平移 ±3cm 步 1cm；旋转 ±10° 步 2.5°
TRANS_GRID = np.linspace(-0.03, 0.03, 7)
ROT_GRID = np.degrees(np.linspace(-10, 10, 9))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    K, gts, depths = z["K"], z["gt"], z["depth"]
    hp = np.load(_os.path.join(_AW, "hand_points.npy"))

    def depth_score(T, depth):
        pc = (T[:3, :3] @ hp.T).T + T[:3, 3]
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

    def scan_axis(build):
        """build(t) -> 4x4；返回沿轴的逐帧平均 score 曲线。"""
        curve = []
        for t in TRANS_GRID if build.__name__ == "t" else ROT_GRID:
            g = np.array([depth_score(build(t, gts[i]), depths[i]) for i in range(len(gts))])
            curve.append(float(g.mean()))
        return curve

    def t(dt, T, ax):
        M = T.copy()
        M[:3, 3] = M[:3, 3] + M[:3, :3] @ (dt * ax)
        return M

    def r(dth, T, ax):
        return T @ rot_about(ax, np.radians(dth))

    out = {}
    print("== GT 邻域扫描(20 帧平均 score，负平均深度残差 m；越接近 0=越贴合) ==", flush=True)
    for ai, ax_name in enumerate("xyz"):
        ax = np.eye(3)[ai]

        def t_b(dt, T, ax=ax):
            return t(dt, T, ax)

        t_b.__name__ = "t"
        c = scan_axis(t_b)
        out[f"trans_{ax_name}"] = dict(grid=TRANS_GRID.tolist(), curve=c)
        best = TRANS_GRID[int(np.argmin(-np.array(c)))]  # score 最大=残差最小
        s_gt, s_best = c[len(c) // 2], max(c)
        print(f"  平移{ax_name}: score(GT)={s_gt*1000:6.2f}mm  峰值@{best*100:+.0f}cm={s_best*1000:6.2f}mm  "
              f"增益={1000*(s_best-s_gt):6.2f}mm  曲线范围={1000*(max(c)-min(c)):6.2f}mm", flush=True)
    for ai, ax_name in enumerate("xyz"):
        ax = np.eye(3)[ai]

        def r_b(dth, T, ax=ax):
            return r(dth, T, ax)

        r_b.__name__ = "r"
        c = scan_axis(r_b)
        out[f"rot_{ax_name}"] = dict(grid=ROT_GRID.tolist(), curve=c)
        best = ROT_GRID[int(np.argmin(-np.array(c)))]
        s_gt, s_best = c[len(c) // 2], max(c)
        print(f"  旋转{ax_name}: score(GT)={s_gt*1000:6.2f}mm  峰值@{best:+.1f}°={s_best*1000:6.2f}mm  "
              f"增益={1000*(s_best-s_gt):6.2f}mm  曲线范围={1000*(max(c)-min(c)):6.2f}mm", flush=True)

    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
