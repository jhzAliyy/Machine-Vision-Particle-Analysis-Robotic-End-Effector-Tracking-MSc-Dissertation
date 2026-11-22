# -*- coding: utf-8 -*-
"""apollo_track_v2.py — 恒定卷跟踪 v2：观测函数换稳健截尾均值。

诊断链(pose_scan / misalign_diag / mask_score)结论：
  原 depth_score 的无截尾均值被背景离群主导(残差 p50≈40mm，p95 达 1-2.5m，
  少量点采到背景深度)——滤波权重的相对比较被离群扭曲。
v2 唯一改动：残差 10% 截尾均值(其余参数/种子/点云与原版逐位一致)。
基线 = 原版 3.2767cm / 23.4560°(自校验一致)。
模型走 apollo_model_fit.npy 缓存(不重辨识)。
产出 $APOLLO_WORK/apollo_track_result_v3.json
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys

import numpy as np

from apollo_track import (ARM_NAMES, B2C, BIAS_BAG, CONST_NPZ, HAND_PTS, INIT_SPREAD,
                          MOVING, NECK, N_P, PAIRS, SIG_MAX, SIG_MIN, T_FRAMES,
                          rot_about, rot_err_deg)
from apollo_overlay import OUT_MODEL, read_bias_bag_once

OUT = _os.path.join(_AW, "apollo_track_result_v3.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    want = set(MOVING) | set(PAIRS[8:]) | {B2C, ("/VICON", "/apollo_head"),
              ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base")}

    print("== 1) 模型(缓存) ==", flush=True)
    model = np.load(OUT_MODEL, allow_pickle=True).item()

    print("== 2) 恒定包单遍读 ==", flush=True)
    j_bias, tf_bias, depths, _ = read_bias_bag_once(BIAS_BAG, want)
    print(f"  深度帧 {len(depths)} | joint_states {len(j_bias)}", flush=True)

    names_b = j_bias[0][1]
    cst = np.load(CONST_NPZ, allow_pickle=False)
    T_hm2bh, T_bh2cam, K = cst["T_hm2bh"], cst["T_bh2cam"], cst["K"]
    pf = json.loads(str(cst["palm_frames"]))
    T_a2p = np.array(pf["/R_ARM_PALM|/R_PALM"])
    hp = np.load(HAND_PTS)
    hp4 = np.concatenate([hp, np.ones((len(hp), 1))], 1) @ np.linalg.inv(T_a2p).T
    hp = hp4[:, :3].astype(np.float32)

    def joint_q(nm, st):
        ts = np.array([x[0] for x in j_bias])
        col = np.array([x[2][names_b.index(nm)] for x in j_bias])
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
        ts = {k: np.array([s for s, _ in tf_bias[k]]) for k in
              [("/VICON", "/apollo_head"), ("/VICON", "/apollo_tool_point")]}

        def at(k):
            i = int(np.argmin(np.abs(ts[k] - st)))
            return tf_bias[k][i][1]

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
        return -float(r[:int(len(r) * 0.9)].mean())          # ← v2 唯一改动：10% 截尾均值

    print("== 3) 跟踪(截尾均值观测，其余同原版) ==", flush=True)
    idx = np.linspace(5, len(depths) - 5, T_FRAMES).astype(int)
    rng = np.random.default_rng(20260924)
    state = rng.normal(0, INIT_SPREAD, (N_P, 7))
    errs, errs0, bhist = [], [], []
    for k, fi in enumerate(idx):
        st, depth = depths[fi]
        base = chain_est(st, {})
        s = np.empty(N_P)
        for i in range(N_P):
            s[i] = depth_score(chain_est(st, dict(zip(ARM_NAMES, state[i]))), depth)
        w = np.exp((s - s.max()) / max(0.5 * s.std(), 1e-6))
        w /= w.sum()
        gt = gt_pose(st)
        est_b = (w[:, None] * state).sum(0)
        est = chain_est(st, dict(zip(ARM_NAMES, est_b)))
        errs.append([np.linalg.norm(est[:3, 3] - gt[:3, 3]) * 100, rot_err_deg(est, gt)])
        errs0.append([np.linalg.norm(base[:3, 3] - gt[:3, 3]) * 100, rot_err_deg(base, gt)])
        bhist.append(np.degrees(est_b))
        pos = (rng.random() + np.arange(N_P)) / N_P
        state = state[np.searchsorted(np.cumsum(w), pos)]
        sig = SIG_MIN + (SIG_MAX - SIG_MIN) * np.exp(-k / 30.0)
        state = state + rng.normal(0, sig, (N_P, 7))
        if (k + 1) % 20 == 0:
            print(f"  帧{k+1}/{T_FRAMES} 平移={errs[-1][0]:.2f}cm 角度={errs[-1][1]:.2f}° "
                  f"b̂={np.round(bhist[-1], 1)}", flush=True)

    errs, errs0 = np.array(errs), np.array(errs0)
    med_t, med_r = np.median(errs, 0)
    med_t0, med_r0 = np.median(errs0, 0)
    print("\n== 4) 判分(100 帧中位) ==", flush=True)
    print(f"  死推算 b=0: 平移={med_t0:.2f}cm 角度={med_r0:.2f}°")
    print(f"  v2 截尾均值跟踪: 平移={med_t:.2f}cm 角度={med_r:.2f}°")
    print(f"  原版基线: 3.2767cm / 23.4560°")
    print(f"  官方对照: FUSION 0.62cm/2.3° | ROTARY 19.7cm/27.0°")
    json.dump(dict(errs=errs.tolist(), errs0=errs0.tolist(),
                   bhat=np.mean(bhist, 0).tolist(),
                   med=[float(med_t), float(med_r), float(med_t0), float(med_r0)]),
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"已存 {OUT}", flush=True)


if __name__ == "__main__":
    main()
