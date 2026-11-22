# -*- coding: utf-8 -*-
"""apollo_track.py — Apollo 恒定偏差考卷全量跟踪测试（v2，零训练，对标官方基准表）。

考卷：synthetic_constant_bias/sime-arm.bag = 真照片 + 篡改读数（每臂关节恒定偏差）+ 动捕真值。
合成包只有动捕/图像/读数三样（模型链被抽掉=防泄漏）；运动学模型全部从 real 包诚实数据辨识。
官方对照（同卷 10 次运行中位）：融合 0.62cm/2.3°、纯编码器 19.7cm/27.0°、纯视觉 47.8cm/79.8°。

机制（本课题的偏差状态化退火进化，run_pf.py 同源）：
  未知量=右臂 7 关节的恒定偏差 b；候选真姿态=FK(读数−b)；打分=深度残差（真 Barrett 掌+前臂点，
  E1-E3/E4 定型的配置：位置配准 2.84mm + 掌块钟差常量 +25° + 指节约定差排除）；
  每帧一代"选择→变异"，抖动退火 2°→0.05°。真裁判=100 帧时序聚合（E 系列判的是更严的逐帧题）。

判据（预先定死）：主判据=手对相机平移中位 ≤1.0cm（贴官方融合 0.62cm 一档）；
  基线价值=比 b=0 死推算低 ≥50%。不达标如实报。
用法：py311 apollo_track.py
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys
import numpy as np
from pathlib import Path

REAL_BAG = Path(_os.path.join(_AW, "real", "bags", "sine-arm.bag"))
BIAS_BAG = Path(_os.path.join(_AW, "synthetic_constant_bias", "bags", "sime-arm.bag"))
CONST_NPZ = _os.path.join(_AW, "apollo_extract.npz")      # sine-arm 常量：T_hm2bh/T_bh2cam/K/palm_frames
HAND_PTS = _os.path.join(_AW, "hand_points.npy")
OUT = _os.path.join(_AW, "apollo_track_result.json")

ARM_NAMES = ["R_SFE", "R_SAA", "R_HR", "R_EB", "R_WR", "R_WFE", "R_WAA"]
PAIRS = [("/BASE", "/R_COLLARBONE"), ("/R_COLLARBONE", "/R_SHOULDER"),
         ("/R_SHOULDER", "/R_UPPER_ARM_IN_SHOULDER"), ("/R_UPPER_ARM_IN_SHOULDER", "/R_UPPER_ARM"),
         ("/R_UPPER_ARM", "/R_FOREARM"), ("/R_FOREARM", "/R_UPPER_WRIST_YAW"),
         ("/R_UPPER_WRIST_YAW", "/R_UPPER_WRIST_PITCH"), ("/R_UPPER_WRIST_PITCH", "/R_LOWER_WRIST"),
         ("/R_LOWER_WRIST", "/R_ARM_PALM"), ("/R_ARM_PALM", "/R_PALM")]
NECK = [("/BASE", "/B_NECK"), ("/B_NECK", "/B_CERVICAL1"), ("/B_CERVICAL1", "/B_CERVICAL2"),
        ("/B_CERVICAL2", "/B_HEAD")]
B2C = ("/B_HEAD", "/XTION_RGB")
MOVING = PAIRS[:8] + NECK

N_P, T_FRAMES = 150, 100
INIT_SPREAD = np.deg2rad(5.0)
SIG_MAX, SIG_MIN = np.deg2rad(2.0), np.deg2rad(0.05)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tf_mat(t):
    T = np.eye(4)
    T[:3, :3] = quat_mat((t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))
    T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
    return T


def rot_axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.array([0, 0, 1.0]), 0.0
    w, v = np.linalg.eig(R)
    a = np.real(v[:, np.argmin(np.abs(w - 1))])
    return a / np.linalg.norm(a), float(ang)


def rot_about(axis, th):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return T


def rot_err_deg(A, B):
    R = A[:3, :3] @ B[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def read_bag(bag, want_pairs, need_depth=False):
    joints, tfseq, depths = [], {k: [] for k in want_pairs}, []
    from rosbags.highlevel import AnyReader
    with AnyReader([bag]) as reader:
        jc = [c for c in reader.connections if c.topic == "/joint_states"]
        tc = [c for c in reader.connections if c.topic == "/tf"]
        for conn, ts, raw in reader.messages(connections=jc):
            m = reader.deserialize(raw, conn.msgtype)
            joints.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, list(m.name), list(m.position)))
        for conn, ts, raw in reader.messages(connections=tc):
            m = reader.deserialize(raw, conn.msgtype)
            for t in m.transforms:
                k = (t.header.frame_id, t.child_frame_id)
                if k in tfseq:
                    tfseq[k].append((t.header.stamp.sec * 10**9 + t.header.stamp.nanosec, tf_mat(t.transform)))
        if need_depth:
            dc = [c for c in reader.connections if c.topic == "/XTION/depth/image"]
            for conn, ts, raw in reader.messages(connections=dc):
                m = reader.deserialize(raw, conn.msgtype)
                depths.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
                               np.frombuffer(m.data, np.float32).reshape(m.height, m.width)))
    return joints, tfseq, depths


def vee_log(R):
    """旋转矩阵→轴角向量（对数映射；符号由矩阵本身决定，稳定不翻）。"""
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang)) * ang


def fit_model(joints, tfseq):
    """模型辨识（real 包诚实数据）：可动对=轴+M 常量+关节名一对一匹配；固定对=均值变换。
    判'可动'必须位置+转角双证据（纯转动件位置失明）；带号转角走对数映射（特征向量符号会翻）。"""
    js = np.array([j[2] for j in joints])
    names = joints[0][1]
    jts = np.array([j[0] for j in joints])
    model = {}
    moving = {}
    for pair in set(MOVING) | set(PAIRS[8:]) | {B2C}:
        seq = tfseq[pair]
        ts = np.array([s for s, _ in seq])
        mats = [m for _, m in seq]
        t_sp = float(max(np.linalg.norm(m[:3, 3] - mats[0][:3, 3]) for m in mats))
        r_sp = float(max(rot_err_deg(m, mats[0]) for m in mats))
        if pair not in MOVING or (t_sp < 1e-4 and r_sp < 0.5):
            model[pair] = dict(name=None, a_c=None, M=np.mean(mats, 0), spread=t_sp)
        else:
            R_rel = mats[-1][:3, :3] @ mats[0][:3, :3].T
            a_p = vee_log(R_rel)
            a_p = a_p / (np.linalg.norm(a_p) + 1e-12)
            steps = np.array([0.0] + [float(np.dot(vee_log(mats[k][:3, :3] @ mats[k - 1][:3, :3].T), a_p))
                                      for k in range(1, len(mats))])
            dR = np.cumsum(steps)
            cand = []
            for ci, nm in enumerate(names):
                q0 = np.interp(ts, jts, js[:, ci])
                q0 = q0 - q0[0]
                for sgn in (+1, -1):
                    cand.append((-float(np.sqrt(np.mean((q0 - sgn * dR) ** 2))), nm, ci, sgn))
            moving[pair] = (ts, mats, a_p, cand)
    # 一对一贪心指派（七臂对不得同抢一名）
    allc = sorted([(c[0], pair, c[1], c[2], c[3]) for pair, (_, _, _, cs) in moving.items() for c in cs],
                  key=lambda x: -x[0])
    assign, used = {}, set()
    for sc, pair, nm, ci, sgn in allc:
        if pair in assign or nm in used:
            continue
        assign[pair] = (nm, ci, sgn, sc)
        used.add(nm)
    for pair, (ts, mats, a_p, _) in moving.items():
        nm, ci, sgn, sc = assign[pair]
        qi = np.interp(ts, jts, js[:, ci])
        a_c = mats[0][:3, :3].T @ (sgn * a_p)
        Ms = [m @ rot_about(a_c, -q) for m, q in zip(mats, qi)]
        M0 = Ms[0]
        # 恒定性体检必须带转角（M 的平移与关节角无关——纯位置指标永远假绿）
        m_sp = float(max(np.linalg.norm(x[:3, 3] - M0[:3, 3]) + 0.01 * np.radians(rot_err_deg(x, M0)) for x in Ms))
        model[pair] = dict(name=nm, a_c=a_c, M=M0, fit=sc, spread=m_sp)
        print(f"  {pair[1]:28s} ↔ {nm:6s} 拟合RMSE={-sc:.4f}rad  M恒定性={m_sp*1000:.2f}(mm+0.01rad混合)", flush=True)
    return model


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    want = set(MOVING) | set(PAIRS[8:]) | {B2C, ("/VICON", "/apollo_head"),
              ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base")}
    print("== 1) 模型辨识（real 包：诚实读数+真 tf） ==", flush=True)
    j_real, tf_real, _ = read_bag(REAL_BAG, want)
    model = fit_model(j_real, tf_real)

    print("== 2) 答案键（两包同录同时间戳，读数直减=注入偏差） ==", flush=True)
    j_bias, tf_bias, depths = read_bag(BIAS_BAG, want, need_depth=True)
    names_r, names_b = j_real[0][1], j_bias[0][1]
    tr = np.array([x[0] for x in j_real])
    tb = np.array([x[0] for x in j_bias])
    deltas = {}
    for nm in ARM_NAMES:
        qb = np.array([x[2][names_b.index(nm)] for x in j_bias])
        qr = np.interp(tb, tr, np.array([x[2][names_r.index(nm)] for x in j_real]))
        deltas[nm] = float(np.mean(qb - qr))
        print(f"  {nm}: 注入偏差={np.degrees(deltas[nm]):+.2f}°", flush=True)

    # 常量与手点
    cst = np.load(CONST_NPZ, allow_pickle=False)
    T_hm2bh, T_bh2cam, K = cst["T_hm2bh"], cst["T_bh2cam"], cst["K"]
    pf = json.loads(str(cst["palm_frames"]))
    T_a2p = np.array(pf["/R_ARM_PALM|/R_PALM"])
    hp = np.load(HAND_PTS)
    hp4 = np.concatenate([hp, np.ones((len(hp), 1))], 1) @ np.linalg.inv(T_a2p).T
    hp = hp4[:, :3].astype(np.float32)   # 统一到官方考核系 R_PALM

    def pair_series(pair):
        seq = tf_bias.get(pair)
        return seq

    def T_pair_const(pair):
        return model[pair]["M"]

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
        return -float(np.mean(np.abs(pc[okz][ing][good, 2] - d_obs[good])))

    print("== 3) 跟踪（偏差状态化退火进化，100 帧） ==", flush=True)
    idx = np.linspace(5, len(depths) - 5, T_FRAMES).astype(int)
    names7 = ARM_NAMES
    rng = np.random.default_rng(20260924)
    state = rng.normal(0, INIT_SPREAD, (N_P, 7))
    errs, errs0, bhist = [], [], []
    for k, fi in enumerate(idx):
        st, depth = depths[fi]
        base = chain_est(st, {})
        s = np.empty(N_P)
        for i in range(N_P):
            s[i] = depth_score(chain_est(st, dict(zip(names7, state[i]))), depth)
        w = np.exp((s - s.max()) / max(0.5 * s.std(), 1e-6))
        w /= w.sum()
        gt = gt_pose(st)
        est_b = (w[:, None] * state).sum(0)
        est = chain_est(st, dict(zip(names7, est_b)))
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
    print("\n== 4) 判分（手对相机位姿误差，官方口径，100 帧中位） ==", flush=True)
    print(f"  死推算 b=0（≈官方 ROTARY 口径）: 平移={med_t0:.2f}cm  角度={med_r0:.2f}°")
    print(f"  带提示跟踪（本课题）:           平移={med_t:.2f}cm  角度={med_r:.2f}°")
    print(f"  官方对照: FUSION 0.62cm/2.3° | ROTARY 19.7cm/27.0° | VISUAL 47.8cm/79.8°")
    print(f"  偏差估计均值 b̂={np.round(np.mean(bhist, 0), 2)}°（答案键={np.round([np.degrees(deltas[n]) for n in names7], 2)}°）")
    c1, c2 = med_t <= 1.0, med_t <= 0.5 * med_t0
    print(f"判据 主判据（平移中位≤1.0cm）: {med_t:.2f}cm → {'通过' if c1 else '不通过'}")
    print(f"判据 基线价值（比死推算低≥50%）: {med_t:.2f} vs {med_t0:.2f}cm → {'通过' if c2 else '不通过'}")
    json.dump(dict(errs=errs.tolist(), errs0=errs0.tolist(), bhat=np.mean(bhist, 0).tolist(),
                   delta_key={n: deltas[n] for n in names7},
                   med=[float(med_t), float(med_r), float(med_t0), float(med_r0)]),
              open(OUT, "w"), ensure_ascii=False)
    print(f"已存 {OUT}")


if __name__ == "__main__":
    main()
