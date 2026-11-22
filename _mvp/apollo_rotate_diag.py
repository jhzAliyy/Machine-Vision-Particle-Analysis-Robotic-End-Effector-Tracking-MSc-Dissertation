# -*- coding: utf-8 -*-
"""apollo_rotate_diag.py — 旋转判别力定点诊断（对照实验，不打补丁）。

三组定点检查，判别五个假设（见对话/归档的假设树）：
  E1 分轴检查（H1）：旋转扰动分 x/y/z 轴各自算逐帧命中率与曲线幅度——
     若某轴≈50% 而其他轴>80%：该轴结构性不可见（滚转盲），混合统计是伪命题。
  E2 零域差对照（H2·关键判别器）：以模型自身在真值位姿的深度渲染当"观测照片"再打分——
     零失配下仍盲=打分器结构性不敏感；锐利可判=打分器无辜，病根在模型与真实照片的失配。
  E3 部件消融（H3/H4）：掌块/手指/前臂分件打分——谁带旋转信号、谁带噪声一目了然。

判读输出：每检查给出支持/反对哪个假设的单向结论，禁止稀泥。
用法：py311 apollo_rotate_diag.py
"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import sys
import numpy as np

NPZ = _os.path.join(_AW, "apollo_extract.npz")   # sine-arm 真实包素材（GT+深度）
HAND = _os.path.join(_AW, "hand_points.npy")
LV_R = (5.0, 10.0, 20.0)   # 旋转扰动（度）


def rot_err_deg(A, B):
    R = A[:3, :3] @ B[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def project(pts, T, K):
    pc = (T[:3, :3] @ pts.T).T + T[:3, 3]
    ok = pc[:, 2] > 0.05
    uv = (K @ pc[ok].T).T
    return uv[:, :2] / uv[:, 2:3], pc[ok][:, 2]


def depth_residual_score(uv, zmod, depth):
    h, w = depth.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ing.sum() < 50:
        return -1.0
    d_obs = depth[v[ing], u[ing]]
    good = np.isfinite(d_obs) & (d_obs > 0.05)
    if good.sum() < 50:
        return -1.0
    return -float(np.mean(np.abs(zmod[ing][good] - d_obs[good])))


def model_depth_map(pts, T, K, shape):
    """z-buffer：模型点在位姿 T 下的深度图（E2 的合成观测）。"""
    zbuf = np.full(shape, np.inf)
    uv, z = project(pts, T, K)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    h, w = shape
    ing = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    np.minimum.at(zbuf, (v[ing], u[ing]), z[ing])
    zbuf[~np.isfinite(zbuf)] = 0.0
    return zbuf


def rot_cands(gt, axis):
    """单轴旋转候选：真值 × ±{5,10,20}°（绕手系原点）。"""
    out = [(0.0, gt.copy())]
    for lvl in LV_R:
        for s in (+1, -1):
            th = np.radians(s * lvl)
            R = np.eye(4)
            c, sn = np.cos(th), np.sin(th)
            a, b = (axis + 1) % 3, (axis + 2) % 3
            R[a, a] = R[b, b] = c
            R[a, b], R[b, a] = -sn, sn
            out.append((np.radians(lvl), gt @ R))
    return out


def evaluate(pts, depths, gts, K, label):
    """分轴评估：每轴曲线 + 大扰动(20°)逐帧命中率。返回 {axis: (curve, hit)}。"""
    res = {}
    for axis in range(3):
        lv, hits, tot = {}, 0, 0
        for i in range(len(gts)):
            scores = [(mag, depth_residual_score(*project(pts, T, K), depths[i]))
                      for mag, T in rot_cands(gts[i], axis)]
            s0 = scores[0][1]
            for mag, s in scores[1:]:
                lv.setdefault(round(np.degrees(mag), 1), []).append(s)
                if round(np.degrees(mag), 1) == 20.0:
                    tot += 1
                    hits += (s0 > s)
        xs = sorted(lv)
        ys = [float(np.mean(lv[x])) for x in xs]
        res[axis] = (list(zip(xs, ys)), hits / max(tot, 1), hits, tot)
        curve = "  ".join(f"{x:.0f}°→{y:.4f}" for x, y in zip(xs, ys))
        print(f"  [{label}] 轴{'xyz'[axis]}: {curve}  |  20°命中={res[axis][1]*100:.1f}% ({hits}/{tot})")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    depth, gt, K = z["depth"], z["gt"], z["K"]
    hp = np.load(HAND)
    pf = json.loads(str(z["palm_frames"]))
    T_fix = np.linalg.inv(np.array(pf["/R_ARM_PALM|/R_PALM"]))
    gts = [g @ T_fix for g in gt]   # 统一到 R_ARM_PALM 挂载系
    n = len(gts)
    print(f"素材 {n} 帧 | 模型点 {len(hp)}（布局：掌 0:400 + 前臂 400:1600；指节已排除）")

    # ---- E1+E2：真实照片 vs 零域差合成（全模型） ----
    print("\n== E1/E2 分轴 × 两种观测（关键判别器：零域差对照） ==", flush=True)
    print("—— 真实深度照片 ——")
    real = evaluate(hp, depth, gts, K, "真实")
    syn_depths = [model_depth_map(hp, T, K, depth[i].shape) for i, T in enumerate(gts)]
    print("—— 模型自渲染深度（零域差）——")
    syn = evaluate(hp, syn_depths, gts, K, "零域差")

    # ---- E3 部件消融（真实照片上分轴 20° 命中，掌/指/前臂组合） ----
    print("\n== E3 部件消融（真实照片，20° 逐帧命中率） ==", flush=True)
    parts = {"仅掌块": hp[:400], "掌+前臂(全件)": hp}   # 指节已排除（约定差 134°），布局=掌400+前臂1200
    abl = {}
    for nm, pts in parts.items():
        abl[nm] = {}
        for axis in range(3):
            hits, tot = 0, 0
            for i in range(n):
                scores = [(mag, depth_residual_score(*project(pts, T, K), depth[i]))
                          for mag, T in rot_cands(gts[i], axis)]
                s0 = scores[0][1]
                for mag, s in scores[1:]:
                    if round(np.degrees(mag), 1) == 20.0:
                        tot += 1
                        hits += (s0 > s)
            abl[nm][axis] = hits / max(tot, 1)
        print(f"  {nm:8s} 轴x={abl[nm][0]*100:5.1f}%  轴y={abl[nm][1]*100:5.1f}%  轴z={abl[nm][2]*100:5.1f}%")

    # ---- 判读（单向结论） ----
    print("\n== 判读 ==", flush=True)
    for axis in range(3):
        hr, hs = real[axis][1], syn[axis][1]
        if hs >= 0.8 and hr < 0.8:
            print(f"  轴{'xyz'[axis]}：零域差 {hs*100:.0f}% 过 / 真实 {hr*100:.0f}% 不过 → 病根=模型与真实照片失配（H3/H4），打分器无辜（反 H2）")
        elif hs < 0.8 and hr < 0.8:
            print(f"  轴{'xyz'[axis]}：零域差也只有 {hs*100:.0f}% → 打分器/几何对该轴结构性不敏感（支持 H2/H1），真实照片非病根")
        elif hs >= 0.8 and hr >= 0.8:
            print(f"  轴{'xyz'[axis]}：双过（零域差 {hs*100:.0f}% / 真实 {hr*100:.0f}%）→ 该轴健康")
        else:
            print(f"  轴{'xyz'[axis]}：真实 {hr*100:.0f}% 过而零域差 {hs*100:.0f}% 不过——异常组合，需复查实现")
    best_part = max(abl, key=lambda k: sum(abl[k].values()))
    print(f"  消融最优部件组合={best_part}（三轴命中和最高）——若'掌+前臂'胜过'全件'，指姿失配（H3）实锤")


if __name__ == "__main__":
    main()
