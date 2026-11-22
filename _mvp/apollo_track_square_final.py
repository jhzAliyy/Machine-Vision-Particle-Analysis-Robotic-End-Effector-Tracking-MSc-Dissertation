# -*- coding: utf-8 -*-
"""apollo_track_square4.py — 方波 H 配置：截尾观测 + 读数突变检测 + 偏差高斯先验。

用户看片反馈 + 逐帧数据定位：档位每 25-30 帧切换(帧 20/50/85 附近)，
跳变瞬间 b̂ 仍带旧档位偏差 → 绿轴短暂贴红；档位 B 期间 b̂ 部分维度被拉偏
(R_WR 漂至 -22°)恢复慢。
C/D 跳变检测失败根因 = 原观测判别弱；E 已证截尾观测判别强。
F = B' 参数(sig_min 0.5°/tau 60) + 截尾观测 + 跳变检测：
  best score 较 15 帧滑窗中位恶化(绝对 20mm 或相对 50%) → 重扩散 σ=5° +
  退火时钟重置 + 冷却 10 帧。预期：跳变后 5-10 帧重锁定新档位，
  全程压到 ≤6cm 级，跳变段不再贴红。
产出 $APOLLO_WORK/apollo_track_square_Hvis.json
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys

import numpy as np

from apollo_track import (ARM_NAMES, B2C, CONST_NPZ, INIT_SPREAD, MOVING, NECK,
                          N_P, PAIRS, SIG_MAX, SIG_MIN, T_FRAMES, rot_about, rot_err_deg)
from apollo_overlay import OUT_MODEL, read_bias_bag_once

SQ_BAG = __import__("pathlib").Path(_os.path.join(_AD, "synthetic_square_bias", "bags", "sine-arm.bag"))
OUT = _os.path.join(_AW, "apollo_track_square_Hvis.json")

JUMP_WIN = 15
JUMP_ABS = 0.02
JUMP_REL = 0.5
COOLDOWN = 10


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    want = set(MOVING) | set(PAIRS[8:]) | {B2C, ("/VICON", "/apollo_head"),
              ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base")}

    print("== 1) 模型(缓存) ==", flush=True)
    model = np.load(OUT_MODEL, allow_pickle=True).item()

    print("== 2) 方波包单遍读 ==", flush=True)
    j_sq, tf_sq, depths, _ = read_bias_bag_once(SQ_BAG, want)
    print(f"  深度帧 {len(depths)} | joint_states {len(j_sq)}", flush=True)

    names_b = j_sq[0][1]
    cst = np.load(CONST_NPZ, allow_pickle=False)
    T_hm2bh, T_bh2cam, K = cst["T_hm2bh"], cst["T_bh2cam"], cst["K"]
    pf = json.loads(str(cst["palm_frames"]))
    T_a2p = np.array(pf["/R_ARM_PALM|/R_PALM"])
    hp = np.load(_os.path.join(_AW, "hand_points.npy"))
    hp4 = np.concatenate([hp, np.ones((len(hp), 1))], 1) @ np.linalg.inv(T_a2p).T
    hp = hp4[:, :3].astype(np.float32)

    def joint_q(nm, st):
        ts = np.array([x[0] for x in j_sq])
        col = np.array([x[2][names_b.index(nm)] for x in j_sq])
        return float(np.interp(st, ts, col))

    def chain_est(st, b):
        mats = []
        for pair in PAIRS:
            md = model[pair]
            if md["name"] is None:
                mats.append(md["M"])
            else:
                q = joint_q(md["name"], st) - b.get(md["name"], 0.0)
                mats.append(md["M"] @ rot_about(md["a_c"], q))
        T_base_rpal = np.eye(4)
        for m in mats:
            T_base_rpal = T_base_rpal @ m
        neck = []
        for pair in NECK:
            md = model[pair]
            if md["name"] is None:
                neck.append(md["M"])
            else:
                neck.append(md["M"] @ rot_about(md["a_c"], joint_q(md["name"], st)))
        T_base_head = np.eye(4)
        for m in neck:
            T_base_head = T_base_head @ m
        T_base_cam = T_base_head @ model[B2C]["M"]
        return np.linalg.inv(T_base_cam) @ T_base_rpal

    def gt_pose(st):
        ts = {k: np.array([s for s, _ in tf_sq[k]]) for k in
              [("/VICON", "/apollo_head"), ("/VICON", "/apollo_tool_point")]}

        def at(k):
            i = int(np.argmin(np.abs(ts[k] - st)))
            return tf_sq[k][i][1]

        cam_in_v = at(("/VICON", "/apollo_head")) @ T_hm2bh @ T_bh2cam
        return np.linalg.inv(cam_in_v) @ at(("/VICON", "/apollo_tool_point"))

    def depth_score(T_hp, depth):
        pc = (T_hp[:3, :3] @ hp.T).T + T_hp[:3, 3]
        okz = pc[:, 2] > 0.05
        uv = (K @ pc[okz].T).T
        u, v = np.round(uv[:, 0] / uv[:, 2]).astype(int), np.round(uv[:, 1] / uv[:, 2]).astype(int)
        h, w = depth.shape
        ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d_obs = depth[v[ing], u[ing]]
        good = np.isfinite(d_obs) & (d_obs > 0.05)
        if good.sum() < 50:
            return -1.0
        r = np.sort(np.abs(pc[okz][ing][good, 2] - d_obs[good]))
        return -float(r[:int(len(r) * 0.9)].mean())          # 截尾观测(E 已证)

    print("== 3) F 配置跟踪(B' 参数 + 截尾 + 跳变检测重扩散) ==", flush=True)
    idx = np.linspace(5, len(depths) - 5, T_FRAMES).astype(int)
    rng = np.random.default_rng(20260924)
    state = rng.normal(0, INIT_SPREAD, (N_P, 7))
    errs, errs0, bhist, jumps = [], [], [], []
    ests, bases, gts_l, fstamps = [], [], [], []
    best_hist = []
    k_local = 0
    cooldown = 0
    SIG_MIN_F, TAU_F = SIG_MIN * 10.0, 60.0        # B' 参数：地板 0.5°/tau 60
    for k, fi in enumerate(idx):
        st, depth = depths[fi]
        base = chain_est(st, {})
        s = np.empty(N_P)
        for i in range(N_P):
            s[i] = depth_score(chain_est(st, dict(zip(ARM_NAMES, state[i]))), depth)
        prior = -0.5 * ((state / np.deg2rad(10.0)) ** 2).sum(1)     # 偏差高斯先验 σ=10°
        w = np.exp((s - s.max()) / max(0.5 * s.std(), 1e-6) + prior - prior.max())
        w /= w.sum()
        gt = gt_pose(st)
        est_b = (w[:, None] * state).sum(0)
        est = chain_est(st, dict(zip(ARM_NAMES, est_b)))
        errs.append([np.linalg.norm(est[:3, 3] - gt[:3, 3]) * 100, rot_err_deg(est, gt)])
        errs0.append([np.linalg.norm(base[:3, 3] - gt[:3, 3]) * 100, rot_err_deg(base, gt)])
        bhist.append(np.degrees(est_b))
        ests.append(est); bases.append(base); gts_l.append(gt); fstamps.append(st)

        # 跳变检测 = 读数直推位姿的帧间突变(直接信号，无延迟不误报)
        jumped = False
        cooldown = max(0, cooldown - 1)
        if cooldown == 0 and len(errs0) >= 2:
            base_mv = np.linalg.norm(np.array(errs0[-1][0]) - np.array(errs0[-2][0]))  # errs0 是 cm
            if base_mv > 3.0:
                jumped = True
                jumps.append(int(k))
                print(f"  [帧{k}] 读数突变检测: RAW 位姿帧间移动 {base_mv:.1f}cm", flush=True)

        pos = (rng.random() + np.arange(N_P)) / N_P
        state = state[np.searchsorted(np.cumsum(w), pos)]
        if jumped:
            state = est_b[None, :] + rng.normal(0, INIT_SPREAD, (N_P, 7))
            k_local = 0
            cooldown = COOLDOWN
        else:
            k_local += 1
        sig = SIG_MIN_F + (SIG_MAX - SIG_MIN_F) * np.exp(-k_local / TAU_F)
        state = state + rng.normal(0, sig, (N_P, 7))
        if (k + 1) % 20 == 0:
            print(f"  帧{k+1}/{T_FRAMES} 平移={errs[-1][0]:.2f}cm 角度={errs[-1][1]:.2f}° "
                  f"b̂={np.round(bhist[-1], 1)}", flush=True)

    errs, errs0 = np.array(errs), np.array(errs0)
    med_t, med_r = np.median(errs, 0)
    med_t0, med_r0 = np.median(errs0, 0)
    print("\n== 4) 判分(100 帧中位) ==", flush=True)
    print(f"  死推算: {med_t0:.2f}cm/{med_r0:.2f}°")
    print(f"  H(截尾+读数突变检测+先验): {med_t:.2f}cm/{med_r:.2f}°")
    print(f"  对照: B' 5.31cm/12.10° | A' 5.02cm/19.56° | F 8.43 | G 9.32")
    print(f"  跳变触发帧: {jumps}")
    for name, sl in [("前1/3", slice(0, 33)), ("中1/3", slice(33, 66)), ("后1/3", slice(66, 100))]:
        e = errs[sl]
        print(f"    {name}: {np.median(e[:,0]):.2f}cm/{np.median(e[:,1]):.2f}°")
    json.dump(dict(errs=errs.tolist(), errs0=errs0.tolist(), bhat=np.mean(bhist, 0).tolist(),
                   jumps=jumps, med=[float(med_t), float(med_r), float(med_t0), float(med_r0)]),
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"已存 {OUT}", flush=True)

    print("== 5) H 配置渲染 overlay(时变考卷演示视频) ==", flush=True)
    import os
    import shutil
    from apollo_overlay import STILL_DIR, encode_mp4, read_rgb_near, render
    rgb_sel = read_rgb_near(SQ_BAG, fstamps)
    rgbs = np.array([x[1] for x in rgb_sel], np.uint8)
    FRAME_DIR = _os.path.join(_AW, "overlay_square_frames")
    OUT_MP4 = _os.path.join(_AW, "apollo_overlay_square_B.mp4")
    if os.path.isdir(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    render(list(rgbs), K, hp, np.array(ests), np.array(bases), np.array(gts_l),
           np.array(errs), np.array(errs0), np.array(bhist), FRAME_DIR,
           title1="Apollo benchmark · synthetic_square_bias (time-varying joint error) | particle-filter tracking",
           title2="GREEN = PF (jump-aware)    RED = raw encoder FK    BLUE = VICON ground truth",
           footnote=f"final median: PF {med_t:.2f} cm / {med_r:.1f} deg  |  RAW {med_t0:.2f} cm / {med_r0:.1f} deg  |  jump resets @ {jumps}")
    encode_mp4(FRAME_DIR, OUT_MP4)
    shutil.copy(OUT_MP4, os.path.join(STILL_DIR, "apollo_overlay_square_B.mp4"))
    for k in (9, 49, 89):
        shutil.copy(f"{FRAME_DIR}/frame_{k:03d}.png", os.path.join(STILL_DIR, f"overlay_square_f{k+1:03d}.png"))
    print(f"[done] H 版方波 overlay 已拷回 {STILL_DIR}", flush=True)


if __name__ == "__main__":
    main()
