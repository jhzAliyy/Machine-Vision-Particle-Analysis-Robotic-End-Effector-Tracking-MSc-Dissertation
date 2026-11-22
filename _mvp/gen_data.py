# -*- coding: utf-8 -*-
"""MVP-M1：PyBullet 生成 UR5 正弦关节轨迹序列（自画自拍，零域差）。

产出 data.npz：
  frames  (300,256,256,3) uint8  观测图像（=渲染图本身）
  q_true  (300,6)          float64 关节角真值（rad）
  ee_true (300,3)          float64 末端位置真值（m）
  meta    JSON 字符串（相机参数/关节索引/预埋偏差说明）
预埋：跟踪器侧另加每关节 ±2° 恒定偏差（在 run_pf.py 里实施，模拟电机读数不准）。
"""
import os
import re
import json
import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
OUT = os.path.join(HERE, "data.npz")
FRAMES, W, H = 300, 256, 256


def find_ur5_urdf():
    """优先用仓库自带的 _mvp/ur5.urdf（可复现基线）；否则在 vendor/ 下找。"""
    local = os.path.join(HERE, "ur5.urdf")
    if os.path.exists(local):
        return local
    for root, dirs, files in os.walk(VENDOR):
        for f in files:
            if f == "ur5.urdf":
                return os.path.join(root, f)
    raise FileNotFoundError("缺 ur5.urdf：把仓库自带 ur5.urdf 放本目录，或放 vendor/**/ur_description/urdf/")


def patch_urdf(src):
    """package:// 前缀改绝对路径（PyBullet 不解析 package://）。网格根=vendor 下含 meshes/ur5 的目录。"""
    mesh_root = None
    for root, dirs, files in os.walk(VENDOR):
        if os.path.basename(root) == "ur5" and os.path.basename(os.path.dirname(root)) == "meshes":
            mesh_root = os.path.dirname(os.path.dirname(root))  # .../ur_description
            break
    if mesh_root is None:
        raise FileNotFoundError("vendor/ 下缺 meshes/ur5 网格（取自 ros-industrial/universal_robot）")
    txt = open(src, encoding="utf-8").read()
    txt = re.sub(r"package://[^/]+/", mesh_root.replace("\\", "/") + "/", txt)
    dst = os.path.join(HERE, "ur5_patched.urdf")
    open(dst, "w", encoding="utf-8").write(txt)
    return dst


def main():
    urdf = patch_urdf(find_ur5_urdf())
    p.connect(p.DIRECT)

    # ---- 白色背景：地板 + 两面墙（大面积白盒，保证红臂轮廓可辨）----
    white = [1.0, 1.0, 1.0, 1.0]
    boxes = [
        ((0.3, 0.3, -0.06), (0, 0, 0), [3, 3, 0.05]),            # 地板
        ((-1.2, 0.3, 1.2), (0, 1.5708, 0), [3, 3, 0.05]),        # 左墙
        ((0.3, 1.8, 1.2), (1.5708, 0, 0), [3, 3, 0.05]),         # 后墙
    ]
    for pos, eul, half in boxes:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=white)
        p.createMultiBody(0, 0, vis, pos, p.getQuaternionFromEuler(eul))

    # ---- UR5 红色涂装（利于直方图打分）----
    try:
        body = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    except p.error:
        # .dae 视觉网格不被接受时：把视觉退化到碰撞网格（.stl）
        txt = open(urdf, encoding="utf-8").read().replace("/visual/", "/collision/").replace(".dae", ".stl")
        open(urdf, "w", encoding="utf-8").write(txt)
        body = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)

    red = [0.85, 0.10, 0.10, 1.0]
    n_joints = p.getNumJoints(body)
    joint_ids, ee_link = [], -1
    for j in range(n_joints):
        info = p.getJointInfo(body, j)
        p.changeVisualShape(body, j, rgbaColor=red)
        if info[2] == p.JOINT_REVOLUTE:
            joint_ids.append(j)
        if info[12].decode() in ("ee_link", "tool0"):
            ee_link = j
    p.changeVisualShape(body, -1, rgbaColor=red)
    joint_ids = joint_ids[:6]
    assert len(joint_ids) == 6, f"应有 6 个转动关节，实得 {len(joint_ids)}"
    if ee_link < 0:
        ee_link = n_joints - 1

    # ---- 固定相机 ----
    view = p.computeViewMatrix([1.15, -1.15, 0.95], [0.35, 0.0, 0.35], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, W / H, 0.05, 4.0)

    # ---- 正弦关节轨迹（各关节频率/相位错开，平滑非平凡运动）----
    q0 = np.array([0.0, -1.1, 1.5, -1.6, -1.57, 0.3])
    amp = np.array([0.35, 0.22, 0.18, 0.30, 0.25, 0.28])
    freq = np.array([1.0, 0.8, 1.1, 0.9, 1.2, 1.0])
    phase = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
    ts = np.arange(FRAMES) / FRAMES
    q_traj = q0[None, :] + amp[None, :] * np.sin(2 * np.pi * freq[None, :] * ts[:, None] + phase[None, :])

    frames = np.empty((FRAMES, W, H, 3), np.uint8)
    ee_true = np.empty((FRAMES, 3))
    for t in range(FRAMES):
        for k, j in enumerate(joint_ids):
            p.resetJointState(body, j, float(q_traj[t, k]))
        _, _, rgb, _, _ = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
        frames[t] = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        ee_true[t] = np.array(p.getLinkState(body, ee_link)[4])
        if t % 30 == 0:
            print(f"[gen_data] frame {t}/{FRAMES}", flush=True)

    meta = dict(joint_ids=joint_ids, ee_link=ee_link, bias_deg=2.0,
                bias_signs=[1, -1, 1, -1, 1, -1],
                note="观测=渲染图本身（零域差）；偏差在跟踪侧施加")
    np.savez_compressed(OUT, frames=frames, q_true=q_traj, ee_true=ee_true, meta=json.dumps(meta))
    print(f"[gen_data] 完成 -> {OUT}  frames={frames.shape} q={q_traj.shape} ee={ee_true.shape}", flush=True)
    p.disconnect()


if __name__ == "__main__":
    main()
