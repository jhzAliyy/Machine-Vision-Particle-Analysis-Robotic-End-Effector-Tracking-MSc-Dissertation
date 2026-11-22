# -*- coding: utf-8 -*-
"""apollo_finger_fix.py — 指节约定旋转推导 + 手指装回 + 判别力三变体对照。

病根（E1-E4 诊断）：JHU 网格系与 Apollo 实测指节系差一个固定约定旋转（消融实测 ≈134° 级别，
挂实测系不修正则三轴全掉分）。本脚本从数据直接解出该旋转：

  X_k = (T_reg25 @ U_k)^(-1) @ M_k      k = 三个指根（L/R KN1、M KN2）
    T_reg25 = 掌块六自由度配准 + 掌块钟差 +25°（E4 双轮标定，与 hand_build2 同源）
    U_k     = JHU URDF 零位下指根 link 系（fk_zero）
    M_k     = 实测指根系（tf 刚性对，palm_frames）
  若 X_k 的旋转部分在三个指根上一致（散布小）→ X = SVD 平均 = 约定旋转；
  深层链段（KN2/KN3 组合）复验同一 X = 独立证据。

三变体判别力对照（真照片+真值，20 帧，零 bag 重读——素材全在 apollo_extract.npz）：
  V0 无指（现行 hand_points.npy）
  V1 模型路径指（T_reg25 @ U_j 挂 JHU 零位指姿，不需要 X）
  V2 实测+约定修正指（M_j @ X^(-1)，用实测链段位姿）
指标：各轴 ±20° 扰动下 score(GT) − max score(扰动) 的均值（判别裕度，越大=朝向越可观测）。

产出（$APOLLO_WORK/）：
  hand_points_fingers.npy   V2 全套点云（掌+前臂+8 指节段）
  apollo_finger_fix.json    X 的轴角/一致性 + 三变体裕度表
  overlay_finger_check.png  一帧投影目检（旧点绿 / 新指黄）
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import os
import sys

import numpy as np
import trimesh

from apollo_hand_build2 import (CHAIN_PAIRS, FINGER_MAP, MESH_DIR, NPZ, REAL_BAG, URDF,
                                fk_zero, fit_rigid6, parse_urdf, read_chain, rpy_mat)
from apollo_track import rot_about, rot_err_deg

OUT_PTS = _os.path.join(_AW, "hand_points_fingers.npy")
OUT_JSON = _os.path.join(_AW, "apollo_finger_fix.json")
OUT_PNG = _os.path.join(_AW, "overlay_finger_check.png")
N_SEG_PTS = 160


def rot_axis_angle_deg(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1e-6:
        return np.array([0, 0, 1.0]), 0.0
    w, v = np.linalg.eig(R)
    a = np.real(v[:, np.argmin(np.abs(w - 1))])
    return a / np.linalg.norm(a), float(ang)


def rot_mean(Rs):
    """SVD 投影平均（近一致旋转群）。"""
    M = np.mean([R for R in Rs], 0)
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    return Vt.T @ D @ U.T


def pad4(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- ① URDF 零位 + 实测指根 + 配准（与 hand_build2 同源复算） ----
    links, joints = parse_urdf(URDF)
    poses = fk_zero(joints)
    z = np.load(NPZ, allow_pickle=False)
    pf = json.loads(str(z["palm_frames"]))
    P = lambda key: np.array(pf[key])
    A_list = [poses["finger_1_prox_link"], poses["finger_2_prox_link"], poses["finger_3_med_link"]]
    B_list = [P("/R_ARM_PALM|/R_HAND_L_FINGER_KNUCKLE_1"),
              P("/R_ARM_PALM|/R_HAND_R_FINGER_KNUCKLE_1"),
              P("/R_ARM_PALM|/R_HAND_M_FINGER_KNUCKLE_2")]
    T_reg, pos_res, conv_res = fit_rigid6(A_list, B_list)
    CLOCK_Z_DEG = 25.0
    Rz = np.eye(4)
    c_, s_ = np.cos(np.radians(CLOCK_Z_DEG)), np.sin(np.radians(CLOCK_Z_DEG))
    Rz[0, 0] = Rz[1, 1] = c_
    Rz[0, 1], Rz[1, 0] = -s_, s_
    T_reg25 = Rz @ T_reg
    print(f"配准复算：位置残差={pos_res*1000:.2f}mm 框架朝向差记录={conv_res:.1f}° 钟差+{CLOCK_Z_DEG:.0f}°", flush=True)

    # ---- ② 约定旋转 X 推导（三指根） ----
    knuckles = [("L", "finger_1_prox_link", "/R_ARM_PALM|/R_HAND_L_FINGER_KNUCKLE_1"),
                ("R", "finger_2_prox_link", "/R_ARM_PALM|/R_HAND_R_FINGER_KNUCKLE_1"),
                ("M", "finger_3_med_link", "/R_ARM_PALM|/R_HAND_M_FINGER_KNUCKLE_2")]
    Xs = []
    for tag, jhu_link, meas_key in knuckles:
        X_k = np.linalg.inv(T_reg25 @ poses[jhu_link]) @ P(meas_key)
        axis, ang = rot_axis_angle_deg(X_k[:3, :3])
        Xs.append(X_k[:3, :3])
        print(f"  指根{tag}: X_k 旋转={ang:7.2f}° 轴={np.round(axis,3)} 平移残差={np.linalg.norm(X_k[:3,3])*1000:.2f}mm", flush=True)
    spread = [rot_err_deg(pad4(A), pad4(B)) for A, B in zip(Xs[1:], Xs[:-1])]
    X_rot = rot_mean(Xs)
    _, X_ang = rot_axis_angle_deg(X_rot)
    print(f"  三指根互散布={np.round(spread,2)}°  X 平均旋转={X_ang:.2f}°（E 系列消融所报约定差 134° 的数据直接解）", flush=True)
    X_inv = np.eye(4)
    X_inv[:3, :3] = X_rot.T

    # ---- ③ 实测链段位姿（/tf 单遍） ----
    print("== 实测指节链读取（real 包 /tf） ==", flush=True)
    chain = read_chain(REAL_BAG, CHAIN_PAIRS)
    C = lambda a, b: chain[(a, b)][0]
    # 各段实测系（R_ARM_PALM 下）：L/R 从 KN1 起，M 从 KN2 起
    M_seg = {}
    for f in "LR":
        k1 = P(f"/R_ARM_PALM|/R_HAND_{f}_FINGER_KNUCKLE_1")
        M_seg[(f, 1)] = k1
        M_seg[(f, 2)] = k1 @ C(f"/R_HAND_{f}_FINGER_KNUCKLE_1", f"/R_HAND_{f}_FINGER_KNUCKLE_2")
        M_seg[(f, 3)] = M_seg[(f, 2)] @ C(f"/R_HAND_{f}_FINGER_KNUCKLE_2", f"/R_HAND_{f}_FINGER_KNUCKLE_3")
    m2 = P("/R_ARM_PALM|/R_HAND_M_FINGER_KNUCKLE_2")
    M_seg[("M", 2)] = m2
    M_seg[("M", 3)] = m2 @ C("/R_HAND_M_FINGER_KNUCKLE_2", "/R_HAND_M_FINGER_KNUCKLE_3")

    # 深层链段复验：KN2/KN3 段的 X 应与指根一致
    deep_checks = []
    seg_link = {("L", 2): "finger_1_med_liink", ("L", 3): "finger_1_dist_link",
                ("R", 2): "finger_2_med_link", ("R", 3): "finger_2_dist_link",
                ("M", 3): "finger_3_dist_link"}
    for key, jhu_link in seg_link.items():
        X_j = np.linalg.inv(T_reg25 @ poses[jhu_link]) @ M_seg[key]
        _, ang = rot_axis_angle_deg(X_j[:3, :3])
        deep_checks.append(float(rot_err_deg(pad4(X_rot), pad4(X_j[:3, :3]))))
        print(f"  链段{key}: X_j 旋转={ang:7.2f}° 与 X 平均差={deep_checks[-1]:.2f}°", flush=True)
    print(f"  深层链段与 X 的偏差：均值={np.mean(deep_checks):.2f}° 最大={np.max(deep_checks):.2f}°", flush=True)

    # ---- ④ 造三变体点云 ----
    rng = np.random.default_rng(20260924)
    hp0 = np.load(_os.path.join(_AW, "hand_points.npy"))           # V0 现行（掌+前臂）
    finger_pts_v1, finger_pts_v2 = [], []                      # V1 模型路径 / V2 实测+X 修正
    for stl, f, seg in FINGER_MAP:
        m = trimesh.load(f"{MESH_DIR}/{stl}")
        pl, _ = trimesh.sample.sample_surface(m, N_SEG_PTS, seed=20260924)
        jhu_link = {"L": {1: "finger_1_prox_link", 2: "finger_1_med_liink", 3: "finger_1_dist_link"},
                    "R": {1: "finger_2_prox_link", 2: "finger_2_med_link", 3: "finger_2_dist_link"},
                    "M": {2: "finger_3_med_link", 3: "finger_3_dist_link"}}[f][seg]
        T_model = T_reg25 @ poses[jhu_link]                     # V1：模型路径（零位指姿）
        T_meas = M_seg[(f, seg)] @ X_inv                        # V2：实测指姿 + 约定修正
        finger_pts_v1.append((T_model[:3, :3] @ pl.T).T + T_model[:3, 3])
        finger_pts_v2.append((T_meas[:3, :3] @ pl.T).T + T_meas[:3, 3])
    fp1 = np.concatenate(finger_pts_v1, 0).astype(np.float32)
    fp2 = np.concatenate(finger_pts_v2, 0).astype(np.float32)
    hp1 = np.concatenate([hp0, fp1], 0)
    hp2 = np.concatenate([hp0, fp2], 0)
    np.save(OUT_PTS, hp2)
    print(f"V2 全套点云 {len(hp2)} 点（掌+前臂 {len(hp0)} + 指 {len(fp2)}）→ {OUT_PTS}", flush=True)

    # ---- ⑤ 判别力对照（真照片+真值，±20° 扰动裕度） ----
    K = z["K"]
    gts, depths, rgbs = z["gt"], z["depth"], z["rgb"]

    def depth_score(pts, T, depth):
        pc = (T[:3, :3] @ pts.T).T + T[:3, 3]
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

    variants = {"V0 无指": hp0, "V1 模型路径指": hp1, "V2 实测+约定修正指": hp2}
    margins = {k: {ax: [] for ax in "xyz"} for k in variants}
    for i in range(len(gts)):
        T_gt, dep = gts[i], depths[i]
        for name, pts in variants.items():
            s0 = depth_score(pts, T_gt, dep)
            for ai, ax in enumerate(np.eye(3)):
                sp = [depth_score(pts, T_gt @ rot_about(ax, s * np.radians(20.0)), dep) for s in (+1, -1)]
                margins[name]["xyz"[ai]].append(s0 - max(sp))
    print("\n== 判别裕度（score(GT) − max score(±20°)，均值，越大=该轴朝向越可观测；单位 m 深度残差） ==", flush=True)
    table = {}
    for name in variants:
        row = {ax: float(np.mean(margins[name][ax])) for ax in "xyz"}
        table[name] = row
        print(f"  {name}: x={row['x']*1000:7.2f}mm  y={row['y']*1000:7.2f}mm  z={row['z']*1000:7.2f}mm", flush=True)

    json.dump(dict(X_axis_angle=[rot_axis_angle_deg(X_rot)[0].tolist(), X_ang],
                   knuckle_angles=[rot_axis_angle_deg(R)[1] for R in Xs],
                   deep_dev_deg=deep_checks, margins=table,
                   n_points=dict(V0=len(hp0), V1=len(hp1), V2=len(hp2))),
              open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    print(f"已存 {OUT_JSON}", flush=True)

    # ---- ⑥ 目检图：一帧投影（旧点绿 / V2 指黄 / GT 轴架蓝） ----
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgbs[10]).convert("RGB")
    d = ImageDraw.Draw(im)
    def proj(pts, T, color, r=1):
        pc = (T[:3, :3] @ pts.T).T + T[:3, 3]
        uv = (K @ pc.T).T
        m = uv[:, 2] > 0.05
        u, v = uv[m, 0] / uv[m, 2], uv[m, 1] / uv[m, 2]
        for x, y in zip(u, v):
            if 0 <= x < im.width and 0 <= y < im.height:
                d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    proj(hp0, gts[10], (60, 230, 60))
    proj(fp2, gts[10], (255, 220, 0), r=2)
    im.save(OUT_PNG)
    print(f"目检图 → {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
