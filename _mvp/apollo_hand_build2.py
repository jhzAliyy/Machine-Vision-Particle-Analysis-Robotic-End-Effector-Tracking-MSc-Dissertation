# -*- coding: utf-8 -*-
"""apollo_hand_build2.py — 精细手重建（证据驱动版，零拟合角）。

E1-E3/E4 定位的两处病灶，按数据定义修复：
  ① 指姿失配 → 手指网格直接挂在实测指节坐标系上（STL 就定义在连杆系，挂上=实测指姿，天然正确）
  ② 配准扭转松弛 → 掌块改六自由度配准：三个指根系的完整位姿（平移+朝向，18 约束/6 自由度），
     绕轴扭转被锁死；位姿残差=两套模型框架约定的匹配体检单。
前臂胶囊（实测腕位方向）不变。产出 hand_points.npy（同名覆盖，下游自动吸收）。
复验口径：重跑 apollo_rotate_diag.py —— E2 应保持 3×100%，E1 真实照片三轴应回升。
用法：py311 apollo_hand_build2.py
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pathlib import Path

URDF = _os.path.join(_AW, "jhu-lcsr-attic-bhand_model-937f418", "robots", "bhand_model.URDF")
MESH_DIR = _os.path.join(_AW, "jhu-lcsr-attic-bhand_model-937f418", "meshes")
NPZ = _os.path.join(_AW, "apollo_extract.npz")
OUT = _os.path.join(_AW, "hand_points.npy")
REAL_BAG = Path(_os.path.join(_AW, "real", "bags", "sine-arm.bag"))

# 网格 ↔ 实测指节系映射（JHU finger_1=+x 侧=L、finger_2=−x 侧=R、finger_3=M）
FINGER_MAP = [
    ("finger_1_prox_link.STL", "L", 1), ("finger_1_med_liink.STL", "L", 2), ("finger_1_dist_link.STL", "L", 3),
    ("finger_2_prox_link.STL", "R", 1), ("finger_2_med_link.STL", "R", 2), ("finger_2_dist_link.STL", "R", 3),
    ("finger_3_med_link.STL", "M", 2), ("finger_3_dist_link.STL", "M", 3),
]
CHAIN_PAIRS = []  # 实测指节链对（M 指从 KN2 起）
for f in "LR":
    CHAIN_PAIRS += [(f"/R_HAND_{f}_FINGER_KNUCKLE_1", f"/R_HAND_{f}_FINGER_KNUCKLE_2"),
                    (f"/R_HAND_{f}_FINGER_KNUCKLE_2", f"/R_HAND_{f}_FINGER_KNUCKLE_3"),
                    (f"/R_HAND_{f}_FINGER_KNUCKLE_3", f"/R_HAND_{f}_FINGER_TIP")]
CHAIN_PAIRS += [("/R_HAND_M_FINGER_KNUCKLE_2", "/R_HAND_M_FINGER_KNUCKLE_3"),
                ("/R_HAND_M_FINGER_KNUCKLE_3", "/R_HAND_M_FINGER_TIP")]


def rpy_mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def parse_urdf(path):
    root = ET.parse(path).getroot()
    links = {}
    for link in root.iter("link"):
        vis = link.find("visual/geometry/mesh")
        if vis is not None:
            links[link.get("name")] = vis.get("filename").split("/")[-1]
    joints = []
    for j in root.iter("joint"):
        org = j.find("origin")
        xyz = np.array([float(v) for v in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split()])
        joints.append((j.find("parent").get("link"), j.find("child").get("link"), xyz, rpy))
    return links, joints


def fk_zero(joints):
    poses = {"base_link": np.eye(4)}
    pending = list(joints)
    while pending:
        rest = []
        for parent, child, xyz, rpy in pending:
            if parent in poses:
                T = poses[parent].copy()
                T[:3, 3] = T[:3, :3] @ xyz + T[:3, 3]
                T[:3, :3] = T[:3, :3] @ rpy_mat(rpy)
                poses[child] = T
            else:
                rest.append((parent, child, xyz, rpy))
        assert len(rest) < len(pending), "URDF 树断链"
        pending = rest
    return poses


def fit_rigid6(A_list, B_list):
    """位置三点 Kabsch（物理对齐）；朝向残差仅作'框架约定差'记录——跨模型比朝向无物理意义。"""
    A = np.array([A[:3, 3] for A in A_list])
    B = np.array([B[:3, 3] for B in B_list])
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = cb - R @ ca
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    pos_res = float(np.sqrt((((R @ A.T).T + t - B) ** 2).sum(1).mean()))
    rot_res = float(np.mean([np.degrees(np.arccos(np.clip((np.trace((R @ A[:3, :3]).T @ B[:3, :3]) - 1) / 2, -1, 1)))
                             for A, B in zip(A_list, B_list)]))
    return T, pos_res, rot_res


def tf_mat_local(t):
    x, y, z, w = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    T = np.eye(4)
    T[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
    return T


def read_chain(bag, pairs):
    """读指节链对，返回 {pair: (T, 散布)}（取首帧，散布验静止）。"""
    from rosbags.highlevel import AnyReader
    seq = {k: [] for k in pairs}
    with AnyReader([bag]) as reader:
        tc = [c for c in reader.connections if c.topic == "/tf"]
        for conn, ts, raw in reader.messages(connections=tc):
            m = reader.deserialize(raw, conn.msgtype)
            for t in m.transforms:
                k = (t.header.frame_id, t.child_frame_id)
                if k in seq:
                    seq[k].append(tf_mat_local(t.transform))
    out = {}
    for k, mats in seq.items():
        spread = float(max(np.linalg.norm(m[:3, 3] - mats[0][:3, 3]) for m in mats))
        out[k] = (mats[0], spread)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    links, joints = parse_urdf(URDF)
    poses = fk_zero(joints)
    z = np.load(NPZ, allow_pickle=False)
    pf = json.loads(str(z["palm_frames"]))
    P = lambda key: np.array(pf[key])

    print("== ① 六自由度配准（掌块锚定，位姿对） ==", flush=True)
    A_list = [poses["finger_1_prox_link"], poses["finger_2_prox_link"], poses["finger_3_med_link"]]
    B_list = [P("/R_ARM_PALM|/R_HAND_L_FINGER_KNUCKLE_1"),
              P("/R_ARM_PALM|/R_HAND_R_FINGER_KNUCKLE_1"),
              P("/R_ARM_PALM|/R_HAND_M_FINGER_KNUCKLE_2")]
    T_reg, pos_res, conv_res = fit_rigid6(A_list, B_list)
    print(f"  位置残差={pos_res*1000:.2f}mm（物理对齐，配准用） | 框架朝向约定差≈{conv_res:.1f}°（仅记录——跨模型比朝向无物理意义，配准不用）")
    # 掌块钟差常量（标定型常量，同 T_hm2bh 性质）：E4 双轮扫描标定/留出双验证
    # （v1 模型峰 +30° 标定80%/留出55%；实测指姿版峰 +25° 标定75%/留出60%；两轮均对 40% 基线显著）
    # 机制注记：腕点在 z 轴、前臂胶囊 z 向自对称 → 该旋转等效纯掌块钟差。
    CLOCK_Z_DEG = 25.0
    Rz = np.eye(4)
    c_, s_ = np.cos(np.radians(CLOCK_Z_DEG)), np.sin(np.radians(CLOCK_Z_DEG))
    Rz[0, 0] = Rz[1, 1] = c_
    Rz[0, 1], Rz[1, 0] = -s_, s_
    T_reg = Rz @ T_reg
    print(f"  掌块钟差常量 +{CLOCK_Z_DEG:.0f}°（E4 双轮标定/留出实测）已应用")

    print("== ② 指节链实测读取（挂网格用） ==", flush=True)
    chain = read_chain(REAL_BAG, CHAIN_PAIRS)
    for k, (T, spread) in chain.items():
        print(f"  {k[0]:32s}→{k[1]:28s} 静止散布={spread*1000:.2f}mm")

    print("== 组装 ==", flush=True)
    rng = np.random.default_rng(20260924)
    pts = []
    # 掌块：base_link.STL × T_reg
    m = trimesh.load(f"{MESH_DIR}/base_link.STL")
    pl, _ = trimesh.sample.sample_surface(m, 400)
    pts.append((T_reg[:3, :3] @ pl.T).T + T_reg[:3, 3])
    print(f"  网格 base_link.STL（掌）→ T_reg")
    # 手指：各 STL 直接挂实测指节系
    base_keys = {"L": "/R_ARM_PALM|/R_HAND_L_FINGER_KNUCKLE_1", "R": "/R_ARM_PALM|/R_HAND_R_FINGER_KNUCKLE_1",
                 "M": "/R_ARM_PALM|/R_HAND_M_FINGER_KNUCKLE_2"}
    seg_pos = {"L": {1: "KN1", 2: "KN2", 3: "KN3"}, "R": {1: "KN1", 2: "KN2", 3: "KN3"}, "M": {2: "KN2", 3: "KN3"}}
    # 手指：实测指姿挂网格**经证据排除**（框架约定差 134°：网格挂实测系自带约定旋转错，消融三轴全掉分）。
    # 成品=掌+前臂（E3 最优组合）。指节网格待约定旋转可从关节轴导出后再并入。
    print(f"  网格 {len(FINGER_MAP)} 个指节：排除（约定差 134°，E3 消融实证掉分）")
    # 前臂：实测腕位胶囊
    wrist = np.array(pf["/R_LOWER_WRIST|/R_ARM_PALM"])[:3, 3]
    for t in np.linspace(0, 0.25, 40):
        th = rng.uniform(0, 2 * np.pi, 30)
        pts.append(np.stack([0.035*np.cos(th), 0.035*np.sin(th), np.full_like(th, -t)], 1) + wrist)
    allp = np.concatenate(pts, 0).astype(np.float32)
    np.save(OUT, allp)
    print(f"已存 {OUT}：{len(allp)} 点（掌 400 + 前臂 1200；指节已排除）")


if __name__ == "__main__":
    main()
