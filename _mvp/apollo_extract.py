# -*- coding: utf-8 -*-
"""apollo_extract.py — 从 Apollo bag 抽取抽查素材并组装真值（只读 bag，写一个 npz）。

组装逻辑（全部来自数据自带观测，零训练）：
1) 抽 N 帧（RGB + 深度 + 时间戳）
2) 同步 /tf：模型链关键对（BASE→…→R_ARM_PALM、BASE→…→B_HEAD、B_HEAD→XTION_RGB、R_ARM_PALM→R_PALM）
   与动捕刚体（/VICON→/apollo_head、/VICON→/apollo_tool_point）——wiki：动捕手架即模型 R_PALM 系
3) 标定：头标刚架(apollo_head)与模型头(B_HEAD)的静态偏移 = 同一刚体双观测的平均相对位姿
   （等价于作者的"离线标定"，标定常量，非学习参数）
4) 真值手对相机：gt_hand_in_cam = cam_in_vicon^-1 @ tool_in_vicon
   其中 cam_in_vicon = head_marker_in_vicon @ T_hm→Bhead @ T_Bhead→XTION_RGB
5) 模型链手对相机（ROTARY 口径，对照官方 stats.txt 的 ROTARY 列）：
   rotary_hand_in_cam = cam_model_in_vicon^-1 @ palm_model_in_vicon（全由 tf 链合成）
6) 输出 apollo_extract.npz；打印 rotary-vs-GT 平移/角度误差统计

用法：py311 apollo_extract.py <bag路径> [帧数=20]
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import sys
import json
import numpy as np


def quat_mat(q):
    """q=(x,y,z,w) → 3x3"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf_to_mat(t):
    T = np.eye(4)
    T[:3, :3] = quat_mat((t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))
    T[:3, 3] = (t.translation.x, t.translation.y, t.translation.z)
    return T


def rot_err_deg(A, B):
    R = A[:3, :3] @ B[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    bag = Path(sys.argv[1])
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # 关键 tf 对：模型链与相机
    want = {("/BASE", "/R_COLLARBONE"), ("/R_COLLARBONE", "/R_SHOULDER"),
            ("/R_SHOULDER", "/R_UPPER_ARM_IN_SHOULDER"), ("/R_UPPER_ARM_IN_SHOULDER", "/R_UPPER_ARM"),
            ("/R_UPPER_ARM", "/R_FOREARM"), ("/R_FOREARM", "/R_UPPER_WRIST_YAW"),
            ("/R_UPPER_WRIST_YAW", "/R_UPPER_WRIST_PITCH"), ("/R_UPPER_WRIST_PITCH", "/R_LOWER_WRIST"),
            ("/R_LOWER_WRIST", "/R_ARM_PALM"), ("/R_ARM_PALM", "/R_PALM"),
            ("/BASE", "/B_NECK"), ("/B_NECK", "/B_CERVICAL1"), ("/B_CERVICAL1", "/B_CERVICAL2"),
            ("/B_CERVICAL2", "/B_HEAD"), ("/B_HEAD", "/XTION_RGB"),
            ("/VICON", "/apollo_head"), ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base"),
            ("/R_ARM_PALM", "/R_PALM"), ("/R_LOWER_WRIST", "/R_ARM_PALM"), ("/R_ARM_PALM", "/R_KUKA_TOOL"),
            ("/R_ARM_PALM", "/R_HAND_M_FINGER_KNUCKLE_2"), ("/R_ARM_PALM", "/R_HAND_L_FINGER_KNUCKLE_1"),
            ("/R_ARM_PALM", "/R_HAND_R_FINGER_KNUCKLE_1")}

    pair_ts = {}       # pair → [(stamp, T)]（/tf 每条消息只带一对，各对频率不同，须逐对同步）
    rgb_frames = []    # (stamp, img)
    depth_frames = []
    cam_k = None

    with AnyReader([bag]) as reader:
        conns = reader.connections
        tfc = [c for c in conns if c.topic == "/tf"]
        rgbc = [c for c in conns if c.topic == "/XTION/rgb/image_color"]
        depc = [c for c in conns if c.topic == "/XTION/depth/image"]
        kc = [c for c in conns if c.topic == "/XTION/rgb/camera_info"]

        for conn, ts, raw in reader.messages(connections=tfc):
            msg = reader.deserialize(raw, conn.msgtype)
            for t in msg.transforms:
                key = (t.header.frame_id, t.child_frame_id)
                if key in want:
                    stamp = t.header.stamp.sec * 10**9 + t.header.stamp.nanosec
                    pair_ts.setdefault(key, []).append((stamp, tf_to_mat(t.transform)))

        for conn, ts, raw in reader.messages(connections=rgbc):
            msg = reader.deserialize(raw, conn.msgtype)
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            rgb_frames.append((msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec, img))

        for conn, ts, raw in reader.messages(connections=depc):
            msg = reader.deserialize(raw, conn.msgtype)
            d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
            depth_frames.append((msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec, d))

        for conn, ts, raw in reader.messages(connections=kc):
            msg = reader.deserialize(raw, conn.msgtype)
            cam_k = np.array(msg.K, float).reshape(3, 3)
            break

    pair_arr = {k: (np.array([s for s, _ in v]), [T for _, T in v]) for k, v in pair_ts.items()}
    want_req = {("/VICON", "/apollo_head"), ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base")}
    assert not (want_req - set(pair_arr)), f"缺必备 tf 对: {want_req - set(pair_arr)}"
    model_ok = not ((want - want_req) - set(pair_arr))
    keys = want if model_ok else want_req
    print(f"tf 对 {len(pair_arr)} 种（模型链{'齐全' if model_ok else '缺失——合成包只留动捕+图像+读数，防泄漏设计'}）"
          f" | RGB {len(rgb_frames)} | 深度 {len(depth_frames)} | K 对角线={np.diag(cam_k)}")

    def get_tf(key, st):
        stamps_k, mats = pair_arr[key]
        i = int(np.argmin(np.abs(stamps_k - st)))
        assert abs(stamps_k[i] - st) < 5e7, (f"{key} 同步失败 差={abs(stamps_k[i]-st)/1e6:.1f}ms "
                                             f"st={st} 覆盖=[{stamps_k[0]}..{stamps_k[-1]}] n={len(stamps_k)}")
        return mats[i]

    # 均匀抽帧并逐对同步最近 tf
    # 公共覆盖窗（模型链 tf 比图像晚起 ~0.5s，首尾帧剔除）
    lo = max(s[0][0] for s in pair_arr.values()) + int(5e7)
    hi = min(s[0][-1] for s in pair_arr.values()) - int(5e7)
    cand = [i for i in range(len(rgb_frames)) if lo <= rgb_frames[i][0] <= hi]
    assert len(cand) >= n_frames, f"公共窗内帧不足: {len(cand)}"
    pick = np.linspace(0, len(cand) - 1, n_frames).astype(int)
    idx = [cand[i] for i in pick]
    rgb_sel, dep_sel, tf_sel, stamps = [], [], [], []
    dep_map = dict(depth_frames)
    for i in idx:
        st, img = rgb_frames[i]
        dk = min(dep_map, key=lambda s: abs(s - st))
        assert abs(dk - st) < 5e7
        rgb_sel.append(img)
        dep_sel.append(dep_map[dk])
        tf_sel.append({k: get_tf(k, st) for k in keys})
        stamps.append(st)

    def chain(d, pairs):
        T = np.eye(4)
        for pr in pairs:
            T = T @ d[pr]
        return T

    neck = [("/BASE", "/B_NECK"), ("/B_NECK", "/B_CERVICAL1"), ("/B_CERVICAL1", "/B_CERVICAL2"),
            ("/B_CERVICAL2", "/B_HEAD")]
    arm = [("/BASE", "/R_COLLARBONE"), ("/R_COLLARBONE", "/R_SHOULDER"),
           ("/R_SHOULDER", "/R_UPPER_ARM_IN_SHOULDER"), ("/R_UPPER_ARM_IN_SHOULDER", "/R_UPPER_ARM"),
           ("/R_UPPER_ARM", "/R_FOREARM"), ("/R_FOREARM", "/R_UPPER_WRIST_YAW"),
           ("/R_UPPER_WRIST_YAW", "/R_UPPER_WRIST_PITCH"), ("/R_UPPER_WRIST_PITCH", "/R_LOWER_WRIST"),
           ("/R_LOWER_WRIST", "/R_ARM_PALM"), ("/R_ARM_PALM", "/R_PALM")]

    # 标定：头标刚架→模型头（同一刚体双观测的平均相对位姿）；模型链缺失时取 sine-arm 常量
    if model_ok:
        T_hm2bh_list = []
        for d in tf_sel:
            head_marker = d[("/VICON", "/apollo_head")]          # VICON→头标
            base_marker = d[("/VICON", "/apollo_base")]          # VICON→基座标
            model_head_in_v = base_marker @ chain(d, neck)       # 模型头在 VICON
            T_hm2bh_list.append(np.linalg.inv(head_marker) @ model_head_in_v)
        T_hm2bh = T_hm2bh_list[0].copy()  # 先用首帧，再查散布
        trans_spread = float(np.mean([np.linalg.norm(T[:3, 3] - T_hm2bh[:3, 3]) for T in T_hm2bh_list]))
        rot_spread = float(np.mean([rot_err_deg(T, T_hm2bh) for T in T_hm2bh_list]))
        print(f"标定 T_头标→模型头: 平移散布={trans_spread*1000:.2f}mm 角度散布={rot_spread:.2f}°（双观测一致性）")
        T_bh2cam = tf_sel[0][("/B_HEAD", "/XTION_RGB")]
    else:
        base_npz = np.load(_os.path.join(_AW, "apollo_extract_sine-arm.npz"), allow_pickle=False)
        T_hm2bh, T_bh2cam = base_npz["T_hm2bh"], base_npz["T_bh2cam"]
        print("标定/相机外参 = 取自 sine-arm 常量（本包无模型链）")
    gt_poses, rotary_poses = [], []
    for d in tf_sel:
        tool = d[("/VICON", "/apollo_tool_point")]           # 动捕手架=R_PALM 系（wiki 原话）
        head_marker = d[("/VICON", "/apollo_head")]
        cam_in_v = head_marker @ T_hm2bh @ T_bh2cam          # 相机在 VICON（真值路）
        gt_poses.append(np.linalg.inv(cam_in_v) @ tool)
        if model_ok:
            base_marker = d[("/VICON", "/apollo_base")]
            cam_model_in_v = base_marker @ chain(d, neck) @ T_bh2cam   # 相机（模型链路=ROTARY 口径）
            palm_model_in_v = base_marker @ chain(d, arm)
            rotary_poses.append(np.linalg.inv(cam_model_in_v) @ palm_model_in_v)
        else:
            rotary_poses.append(np.eye(4))

    gt_poses = np.array(gt_poses)
    rotary_poses = np.array(rotary_poses)
    if model_ok:
        t_err = [np.linalg.norm(a[:3, 3] - b[:3, 3]) * 100 for a, b in zip(rotary_poses, gt_poses)]
        r_err = [rot_err_deg(a, b) for a, b in zip(rotary_poses, gt_poses)]
        print(f"ROTARY 口径 vs 真值（本 bag {n_frames} 帧）：平移  中位={np.median(t_err):.2f}cm 均值={np.mean(t_err):.2f}cm | "
              f"角度  中位={np.median(r_err):.2f}° 均值={np.mean(r_err):.2f}°")
        print("（官方 stats.txt 的 ROTARY 列：synthetic_constant_bias 平移中位 19.72cm/角度中位 26.99°；"
              "real 包编码器准应显著更小——本值即 neck+手架误差地板）")
    else:
        print("（合成包无模型链，rotary 口径不适用——死推算基线由 apollo_track.py 的 b=0 给出）")

    # 掌部刚性几何（R_PALM/R_KUKA_TOOL/指根/腕，均为掌系常量；散布验刚性）
    if model_ok:
        rigid_keys = [("/R_ARM_PALM", "/R_PALM"), ("/R_LOWER_WRIST", "/R_ARM_PALM"), ("/R_ARM_PALM", "/R_KUKA_TOOL"),
                      ("/R_ARM_PALM", "/R_HAND_M_FINGER_KNUCKLE_2"), ("/R_ARM_PALM", "/R_HAND_L_FINGER_KNUCKLE_1"),
                      ("/R_ARM_PALM", "/R_HAND_R_FINGER_KNUCKLE_1")]
        palm_frames = {}
        for k in rigid_keys:
            mats = [d[k] for d in tf_sel]
            T0 = mats[0]
            spread = float(max(np.linalg.norm(T[:3, 3] - T0[:3, 3]) for T in mats))
            assert spread < 2e-3, f"{k} 非刚性? 散布={spread*1000:.2f}mm"
            palm_frames[k] = T0
            print(f"  刚性对 {k[0]}→{k[1]}: 散布<{spread*1000:.3f}mm 平移={np.round(T0[:3,3],4)}")
    else:
        palm_frames = json.loads(str(np.load(_os.path.join(_AW, "apollo_extract_sine-arm.npz"),
                                             allow_pickle=False)["palm_frames"]))
        print("掌部刚性几何 = 取自 sine-arm 常量")

    out = _os.path.join(_AW, "apollo_extract_{bag.stem}.npz")
    np.savez_compressed(out, rgb=np.array(rgb_sel, np.uint8), depth=np.array(dep_sel, np.float32),
                        gt=gt_poses, rotary=rotary_poses, stamps=np.array(stamps),
                        T_hm2bh=T_hm2bh, T_bh2cam=T_bh2cam, K=cam_k,
                        palm_frames=json.dumps({f"{a}|{b}": T.tolist() for (a, b), T in palm_frames.items()}))
    print(f"已存 {out}")


if __name__ == "__main__":
    main()
