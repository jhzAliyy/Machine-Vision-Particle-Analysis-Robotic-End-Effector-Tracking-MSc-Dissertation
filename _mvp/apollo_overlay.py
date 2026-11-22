# -*- coding: utf-8 -*-
"""apollo_overlay.py — 导师点名的 overlay 视图：在 Apollo 真视频上叠加末端执行器位姿标注。

产出（$APOLLO_WORK/）：
  apollo_overlay.npz      每帧 RGB + 三套手部位姿（PF 估计 / 盲信读数死推算 / 动捕真值）+ 误差序列
  apollo_model_fit.npy    模型辨识缓存（real 包辨识结果，供后续脚本复用，免重读 1.9GB bag）
  overlay_frames/*.png    逐帧叠加图
  apollo_overlay.mp4      叠加视频（8 fps）
  （mp4 与三张代表帧静图同步拷回 _mvp/apollo_out/）

保真设计：跟踪段与 apollo_track.py 逐位同源——同 rng 种子 20260924、同 N_P/T_FRAMES/退火参数、
chain_est/depth_score/gt_pose 原文复制（rng 消耗顺序零改动）→ 跑完与 apollo_track_result.json
的 med 四元组自校验（容差 1e-6，不符即大声报警）。画图层与数据源无关：帧+位姿+K 三样即可。

画面约定：绿=粒子滤波修正估计；红=盲信读数死推算（b=0）；蓝=VICON 动捕真值（三轴架）。
点的云=Barrett 手+前臂点集在各位姿下的投影；三色轴架=各位姿坐标系的 x/y/z 轴（10cm）。
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from apollo_track import (ARM_NAMES, B2C, BIAS_BAG, CONST_NPZ, HAND_PTS, INIT_SPREAD, MOVING, NECK,
                          N_P, PAIRS, REAL_BAG, SIG_MAX, SIG_MIN, T_FRAMES, fit_model, read_bag,
                          rot_about, rot_err_deg, tf_mat)

WORK = _AW
OUT_NPZ = f"{WORK}/apollo_overlay.npz"
OUT_MODEL = f"{WORK}/apollo_model_fit.npy"
FRAME_DIR = f"{WORK}/overlay_frames"
OUT_MP4 = f"{WORK}/apollo_overlay.mp4"
STILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apollo_out")

names7 = ARM_NAMES


def read_bias_bag_once(bag, want_pairs):
    """单遍读 bias 包：/tf + /joint_states + 深度全存；RGB 只记时间戳（图像第二轮按需取，省内存）。"""
    joints, tfseq, depths, rgb_stamps = [], {k: [] for k in want_pairs}, [], []
    from rosbags.highlevel import AnyReader
    with AnyReader([bag]) as reader:
        keep = [c for c in reader.connections if c.topic in
                ("/tf", "/joint_states", "/XTION/depth/image", "/XTION/rgb/image_color")]
        for conn, ts, raw in reader.messages(connections=keep):
            if conn.topic == "/tf":
                m = reader.deserialize(raw, conn.msgtype)
                for t in m.transforms:
                    k = (t.header.frame_id, t.child_frame_id)
                    if k in tfseq:
                        tfseq[k].append((t.header.stamp.sec * 10**9 + t.header.stamp.nanosec,
                                         tf_mat(t.transform)))
            elif conn.topic == "/joint_states":
                m = reader.deserialize(raw, conn.msgtype)
                joints.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
                               list(m.name), list(m.position)))
            elif conn.topic == "/XTION/depth/image":
                m = reader.deserialize(raw, conn.msgtype)
                depths.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
                               np.frombuffer(m.data, np.float32).reshape(m.height, m.width)))
            else:
                m = reader.deserialize(raw, conn.msgtype)
                rgb_stamps.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)
    return joints, tfseq, depths, np.array(rgb_stamps)


def read_rgb_near(bag, target_stamps, tol_ns=2.5e7):
    """第二轮只取离目标时间戳最近的 RGB 图（每目标一张，容差 25ms）。"""
    best = {}
    tgt = np.asarray(target_stamps)
    from rosbags.highlevel import AnyReader
    with AnyReader([bag]) as reader:
        rc = [c for c in reader.connections if c.topic == "/XTION/rgb/image_color"]
        for conn, ts, raw in reader.messages(connections=rc):
            m = reader.deserialize(raw, conn.msgtype)
            st = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            i = int(np.argmin(np.abs(tgt - st)))
            dist = abs(tgt[i] - st)
            if dist < tol_ns and (i not in best or dist < best[i][0]):
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                best[i] = (dist, st, img.copy())
    missing = [i for i in range(len(tgt)) if i not in best]
    assert not missing, f"RGB 同步失败的目标帧: {missing}"
    return [best[i][1:] for i in range(len(tgt))]


def track_phase(joints, tf_bias, depths, model, T_hm2bh, T_bh2cam, K, hp):
    """跟踪主循环——与 apollo_track.py main 逐位同源（rng 消耗顺序零改动），仅增加位姿/RGB 存储。"""
    names_b = joints[0][1]
    j_bias = joints

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
        return -float(r[:int(len(r) * 0.9)].mean())          # v3 观测：10% 截尾均值（朝向 23.46°→9.19°）

    idx = np.linspace(5, len(depths) - 5, T_FRAMES).astype(int)
    rng = np.random.default_rng(20260924)
    state = rng.normal(0, INIT_SPREAD, (N_P, 7))
    errs, errs0, bhist, ests, bases, gts, frame_stamps = [], [], [], [], [], [], []
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
        ests.append(est)
        bases.append(base)
        gts.append(gt)
        frame_stamps.append(st)
        pos = (rng.random() + np.arange(N_P)) / N_P
        state = state[np.searchsorted(np.cumsum(w), pos)]
        sig = SIG_MIN + (SIG_MAX - SIG_MIN) * np.exp(-k / 30.0)
        state = state + rng.normal(0, sig, (N_P, 7))
        if (k + 1) % 20 == 0:
            print(f"  帧{k+1}/{T_FRAMES} 平移={errs[-1][0]:.2f}cm 角度={errs[-1][1]:.2f}° "
                  f"b̂={np.round(bhist[-1], 1)}", flush=True)
    return (np.array(errs), np.array(errs0), np.array(bhist), np.array(ests),
            np.array(bases), np.array(gts), np.array(frame_stamps))


# ---------------- 画图层（与数据源无关：帧 + 位姿 + K 即可） ----------------

def _project(K, pts):
    """3D 点 → 像素坐标与有效掩码（z>0.05m）。"""
    uvw = (K @ pts.T).T
    ok = uvw[:, 2] > 0.05
    u, v = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-9), uvw[:, 1] / np.maximum(uvw[:, 2], 1e-9)
    return u, v, ok


def _inb(u, v, w, h):
    return (u >= 0) & (u < w) & (v >= 0) & (v < h)


def draw_axes(d, K, T, color, width, w, h, label=None, font=None):
    """位姿坐标系三轴架（原点 + x/y/z 三轴 10cm）。"""
    c = T[:3, 3]
    R = T[:3, :3]
    ends = np.stack([c + 0.10 * R[:, i] for i in range(3)])
    u0, v0, ok0 = _project(K, c[None, :])
    u1, v1, ok1 = _project(K, ends)
    if not (ok0[0] and ok1.all()):
        return
    if not (_inb(np.array([u0[0]]), np.array([v0[0]]), w, h)[0]):
        return
    for i in range(3):
        x2, y2 = u1[i], v1[i]
        if not (0 <= x2 < w and 0 <= y2 < h):
            continue
        d.line([(u0[0], v0[0]), (x2, y2)], fill=color, width=width)
    r = 5
    d.ellipse([u0[0] - r, v0[0] - r, u0[0] + r, v0[0] + r], outline=color, width=3)
    if label and font is not None:
        d.text((u0[0] + 8, v0[0] - 26), label, fill=color, font=font)


def draw_cloud(d, K, hp, T, color, w, h):
    """手+前臂点集在位姿 T 下的投影（1px 点云）。"""
    pc = (T[:3, :3] @ hp.T).T + T[:3, 3]
    u, v, ok = _project(K, pc)
    m = ok & _inb(u, v, w, h)
    if m.sum() == 0:
        return
    d.point(list(zip(u[m].tolist(), v[m].tolist())), fill=color)


def render(rgb_list, K, hp, ests, bases, gts, errs, errs0, bhist, frame_dir,
           title1=None, title2=None, footnote=None):
    from PIL import Image, ImageDraw, ImageFont
    def _font(name, size):
        fp = _os.path.join(_os.environ.get("WINDIR", "C:/Windows"), "Fonts", name)
        return ImageFont.truetype(fp, size) if _os.path.exists(fp) else ImageFont.load_default()
    F_T = _font("consolab.ttf", 17)
    F_S = _font("consola.ttf", 15)
    F_AX = _font("consolab.ttf", 16)
    GREEN, RED, BLUE, WHITE = (60, 255, 60), (255, 60, 60), (90, 150, 255), (245, 245, 245)
    n = len(rgb_list)
    h0, w0 = rgb_list[0].shape[:2]
    os.makedirs(frame_dir, exist_ok=True)
    run_med, run_med0 = [], []
    for k in range(n):
        run_med.append(np.median(errs[:k + 1, 0]))
        run_med0.append(np.median(errs0[:k + 1, 0]))
    for k in range(n):
        im = Image.fromarray(rgb_list[k]).convert("RGB")
        d = ImageDraw.Draw(im)
        # 点云：红（盲信读数）垫底，绿（PF 估计）盖上
        draw_cloud(d, K, hp, bases[k], RED, w0, h0)
        draw_cloud(d, K, hp, ests[k], GREEN, w0, h0)
        # 轴架：真值蓝、盲信红、估计绿
        draw_axes(d, K, gts[k], BLUE, 4, w0, h0, label="GT (VICON)", font=F_AX)
        draw_axes(d, K, bases[k], RED, 3, w0, h0, label="RAW", font=F_AX)
        draw_axes(d, K, ests[k], GREEN, 3, w0, h0, label="PF", font=F_AX)
        # 半透明信息条
        ov = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
        do = ImageDraw.Draw(ov)
        do.rectangle([0, 0, w0, 44], fill=(10, 10, 10, 175))
        do.rectangle([0, h0 - 118, w0, h0], fill=(10, 10, 10, 175))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(im)
        d.text((10, 4), title1 or "Apollo benchmark · synthetic_constant_bias | particle-filter end-effector tracking",
               font=F_T, fill=WHITE)
        d.text((10, 24), title2 or "GREEN = PF estimate    RED = raw encoder FK (no correction)    BLUE = VICON ground truth",
               font=F_S, fill=(210, 210, 210))
        e = ests[k][:3, 3] * 100.0
        y = h0 - 112
        d.text((10, y), f"frame {k+1}/{n}   tool-tip (PF):  x={e[0]:7.2f}  y={e[1]:7.2f}  z={e[2]:7.2f}  cm",
               font=F_S, fill=WHITE)
        d.text((10, y + 22), f"PF  err: {errs[k,0]:6.2f} cm / {errs[k,1]:5.1f} deg"
                             f"      RAW err: {errs0[k,0]:6.2f} cm / {errs0[k,1]:5.1f} deg", font=F_S, fill=WHITE)
        d.text((10, y + 44), f"running median:  PF {run_med[k]:5.2f} cm   |   RAW {run_med0[k]:5.2f} cm",
               font=F_S, fill=WHITE)
        bh = "  ".join(f"{v:5.1f}" for v in np.mean(bhist[:k + 1], 0))
        d.text((10, y + 66), f"joint-bias estimate b-hat (deg): {bh}",
               font=F_S, fill=(200, 255, 200))
        d.text((10, y + 88), footnote or
               "final: PF 3.98 cm / 9.2 deg  |  RAW 21.0 cm / 27.2 deg  |  official ROTARY 19.7 / 27.0",
               font=F_S, fill=(160, 160, 160))
        im.save(f"{frame_dir}/frame_{k:03d}.png")
    return frame_dir


def encode_mp4(frame_dir, out_mp4, fps=8):
    r = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frame_dir}/frame_%03d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
                        "-movflags", "+faststart", out_mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("ffmpeg 编码失败")
    print(f"[mp4] {out_mp4}  {os.path.getsize(out_mp4)/1e6:.1f} MB", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    want = set(MOVING) | set(PAIRS[8:]) | {B2C, ("/VICON", "/apollo_head"),
              ("/VICON", "/apollo_tool_point"), ("/VICON", "/apollo_base")}

    print("== 1) 模型辨识（real 包，缓存复用） ==", flush=True)
    if os.path.exists(OUT_MODEL):
        model = np.load(OUT_MODEL, allow_pickle=True).item()
        print("  载入缓存 apollo_model_fit.npy", flush=True)
    else:
        j_real, tf_real, _ = read_bag(REAL_BAG, want)
        model = fit_model(j_real, tf_real)
        np.save(OUT_MODEL, model, allow_pickle=True)
        print(f"  已缓存 -> {OUT_MODEL}", flush=True)

    print("== 2) bias 包单遍读（tf+读数+深度+RGB时间戳） ==", flush=True)
    j_bias, tf_bias, depths, rgb_stamps = read_bias_bag_once(BIAS_BAG, want)
    print(f"  深度帧 {len(depths)} | RGB 帧 {len(rgb_stamps)} | joint_states {len(j_bias)}", flush=True)

    cst = np.load(CONST_NPZ, allow_pickle=False)
    T_hm2bh, T_bh2cam, K = cst["T_hm2bh"], cst["T_bh2cam"], cst["K"]
    pf = json.loads(str(cst["palm_frames"]))
    T_a2p = np.array(pf["/R_ARM_PALM|/R_PALM"])
    hp = np.load(HAND_PTS)
    hp4 = np.concatenate([hp, np.ones((len(hp), 1))], 1) @ np.linalg.inv(T_a2p).T
    hp = hp4[:, :3].astype(np.float32)   # 统一到官方考核系 R_PALM

    print("== 3) 跟踪（与 apollo_track.py 逐位同源，同种子 20260924） ==", flush=True)
    errs, errs0, bhist, ests, bases, gts, frame_stamps = track_phase(
        j_bias, tf_bias, depths, model, T_hm2bh, T_bh2cam, K, hp)

    med = np.median(errs, 0)
    med0 = np.median(errs0, 0)
    print(f"  PF 中位: {med[0]:.4f}cm / {med[1]:.4f}°   死推算中位: {med0[0]:.4f}cm / {med0[1]:.4f}°", flush=True)

    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "apollo_out", "apollo_track_result.json")))["med"]
    ok = np.allclose(med, ref[:2], atol=1e-6) and np.allclose(med0, ref[2:], atol=1e-6)
    print(f"  自校验 vs apollo_track_result.json med={np.round(ref,4)} → {'一致 ✓' if ok else '!!! 不一致 !!!'}", flush=True)
    if not ok:
        print("  [警告] 复算与存档不一致——渲染继续，但结果需人工核查", flush=True)

    print("== 4) RGB 第二轮按需取图 ==", flush=True)
    rgb_sel = read_rgb_near(BIAS_BAG, frame_stamps)
    print(f"  取到 {len(rgb_sel)} 帧RGB，尺寸 {rgb_sel[0][1].shape}", flush=True)
    rgbs = np.array([x[1] for x in rgb_sel], np.uint8)

    np.savez_compressed(OUT_NPZ, rgb=rgbs, K=K, est=ests, base=bases, gt=gts,
                        errs=errs, errs0=errs0, bhist=bhist, frame_stamps=frame_stamps)
    print(f"  已存 {OUT_NPZ}", flush=True)

    print("== 5) 渲染叠加帧 + 编码 mp4 ==", flush=True)
    if os.path.isdir(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    render(list(rgbs), K, hp, ests, bases, gts, errs, errs0, bhist, FRAME_DIR)
    encode_mp4(FRAME_DIR, OUT_MP4)
    os.makedirs(STILL_DIR, exist_ok=True)
    shutil.copy(OUT_MP4, os.path.join(STILL_DIR, "apollo_overlay.mp4"))
    for k in (9, 49, 89):
        shutil.copy(f"{FRAME_DIR}/frame_{k:03d}.png", os.path.join(STILL_DIR, f"overlay_still_f{k+1:03d}.png"))
    print(f"[done] mp4 + 三张静图已拷回 {STILL_DIR}", flush=True)


if __name__ == "__main__":
    main()
