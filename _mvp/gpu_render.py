# -*- coding: utf-8 -*-
"""gpu_render.py — 真·GPU 光栅化离屏渲染器（moderngl），UR5 场景自绘。

存在理由：pybullet DIRECT 只有 CPU 软渲染（EGL 缺失、硬件 OpenGL 静默回退——
gpu_probe.py 实测两路逐像素一致），150 粒子/帧 4.2~7.6s 压不住实时性。
本文件用显卡光栅化重画；运动学仍由 pybullet 提供（resetJointState + getLinkState），
网格 trimesh 读（.dae 走 pycollada）。只测不写盘，不改动 run_pf.py / gen_data.py。

判据（预先定死，跑完对照）：
  1) GL_RENDERER 是真显卡（含 NVIDIA/Intel/AMD/Radeon/GeForce，非 Microsoft GDI/软件光栅）
  2) 画面含红臂（红像素>0），同姿态红质心与 TinyRenderer 相差 < 画面边长 15%
  3) 帧时 < CPU 基线 128²:28.2ms / 224²:46.9ms（gpu_probe.py 同姿态预热2+计时20 实测）

用法：py311 gpu_render.py
备选：MESH_MODE="collision" 切 .stl 碰撞网格（.dae 读取器有坐标系约定差时用）。
白盒背景（地板+两墙）以白色清屏等价替代：它们在画面里只贡献纯白像素。
"""
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import moderngl
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_MODE = "visual"  # "visual"=.dae 视觉网格 / "collision"=.stl 碰撞网格
RED = (0.85, 0.10, 0.10)
N_WARM, N_TIME = 2, 20
SIZES = (128, 224)
CPU_MS = {128: 28.2, 224: 46.9}  # 判据基线（gpu_probe.py 冻结值）

VS = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_norm;
out vec3 v_norm;
void main() {
    v_norm = mat3(model) * in_norm;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FS = """
#version 330
uniform vec3 color;
in vec3 v_norm;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    float d = 0.55 + 0.45 * max(dot(n, normalize(vec3(0.4, -0.5, 0.75))), 0.0);
    frag = vec4(color * d, 1.0);
}
"""


def parse_urdf_meshes(urdf):
    """读 URDF 每个 link 的 visual（或 collision）网格：(link名, 网格路径, origin_xyz, origin_rpy)。"""
    tag = "visual" if MESH_MODE == "visual" else "collision"
    root = ET.parse(urdf).getroot()
    out = []
    for link in root.iter("link"):
        for node in link.findall(tag):
            mesh = node.find("geometry/mesh")
            if mesh is None:
                continue
            origin = node.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
            out.append((link.get("name"), mesh.get("filename"), xyz, rpy))
    return out


def link_index_map(body):
    m = {p.getBodyInfo(body)[0].decode(): -1}
    for j in range(p.getNumJoints(body)):
        m[p.getJointInfo(body, j)[12].decode()] = j
    return m


def pose_mat(pos, orn):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def link_world(body, link):
    if link < 0:
        return np.eye(4)
    st = p.getLinkState(body, link)
    return pose_mat(st[4], st[5])  # URDF link 坐标系的世界位姿（与 visual origin 同一坐标系）


def load_mesh(path):
    """trimesh 读网格 → 炸开成逐面顶点（平面着色，贴近 Tiny 观感）+ 逐面法线。"""
    m = trimesh.load(path, force="mesh")
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)
    fn = np.asarray(m.face_normals, np.float64)
    tri = v[f].reshape(-1, 3)
    nrm = np.repeat(fn, 3, axis=0)
    return np.hstack([tri, nrm]).astype(np.float32), len(tri), m.extents


class GpuRenderer:
    def __init__(self, body, urdf, joint_ids):
        self.body, self.joint_ids = body, joint_ids
        self.ctx = moderngl.create_standalone_context()
        self.info = self.ctx.info
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.items, self.fbos = [], {}
        lim = link_index_map(body)
        for name, fn, xyz, rpy in parse_urdf_meshes(urdf):
            data, count, ext = load_mesh(fn)
            print(f"  [mesh] {name:14s} {os.path.basename(fn):14s} 顶点={count:6d} 尺寸={np.round(ext, 3)}")
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_norm")])
            local = pose_mat(xyz, p.getQuaternionFromEuler(rpy))
            self.items.append((vao, count, local, lim[name]))

    def _fbo(self, size):
        if size not in self.fbos:
            tex = self.ctx.texture((size, size), 3)
            depth = self.ctx.depth_renderbuffer((size, size))
            self.fbos[size] = self.ctx.framebuffer(color_attachments=[tex], depth_attachment=depth)
        return self.fbos[size]

    def render(self, q, size, view, proj):
        for k, j in enumerate(self.joint_ids):
            p.resetJointState(self.body, j, float(q[k]))
        fbo = self._fbo(size)
        fbo.use()
        self.ctx.viewport = (0, 0, size, size)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)  # 白色背景 ≡ 白盒三件套的画面贡献
        Vm = np.array(view, np.float64).reshape(4, 4).T   # pybullet 16 浮点=列主序 → 转成数学矩阵
        Pm = np.array(proj, np.float64).reshape(4, 4).T
        for vao, count, local, link in self.items:
            model = link_world(self.body, link) @ local
            mvp = Pm @ Vm @ model
            # 上传列主序字节 = 数学矩阵转置后按行铺开
            self.prog["mvp"].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())
            self.prog["model"].write(np.ascontiguousarray(model.T, dtype=np.float32).tobytes())
            self.prog["color"].value = RED
            vao.render()
        raw = fbo.read(components=3)  # GL 原点在左下，需上下翻转
        return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(size, size, 3)[::-1])


def tiny_shoot(q, size, body, joint_ids, view, proj):
    for k, j in enumerate(joint_ids):
        p.resetJointState(body, j, float(q[k]))
    _, _, rgb, _, _ = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.reshape(rgb, (size, size, 4))[:, :, :3].astype(np.uint8)


def red_stats(im):
    r, g, b = (im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int))
    mask = (r > 150) & (g < 100) & (b < 100)
    n = int(mask.sum())
    if n == 0:
        return 0, (float("nan"), float("nan"))
    ys, xs = np.nonzero(mask)
    return n, (float(xs.mean()), float(ys.mean()))


def project_ee(pos, view, proj, size):
    """把真值末端点投到像素平面——与两路红质心对照，锁定相机矩阵没错位。"""
    Vm = np.array(view, np.float64).reshape(4, 4).T
    Pm = np.array(proj, np.float64).reshape(4, 4).T
    clip = Pm @ Vm @ np.array([pos[0], pos[1], pos[2], 1.0])
    ndc = clip[:3] / clip[3]
    return ((ndc[0] + 1) / 2 * size, (1 - ndc[1]) / 2 * size)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    z = np.load(os.path.join(HERE, "data.npz"))
    meta = json.loads(str(z["meta"]))
    joint_ids = meta["joint_ids"]
    poses = z["q_true"][::15][:N_TIME]

    urdf = os.path.join(HERE, "ur5_patched.urdf")
    p.connect(p.DIRECT)
    # Tiny 对照侧：与 run_pf.py 完全一致的场景（白盒三件 + 红臂涂装）
    white = [1.0, 1.0, 1.0, 1.0]
    for pos, eul, half in [((0.3, 0.3, -0.06), (0, 0, 0), [3, 3, 0.05]),
                           ((-1.2, 0.3, 1.2), (0, 1.5708, 0), [3, 3, 0.05]),
                           ((0.3, 1.8, 1.2), (1.5708, 0, 0), [3, 3, 0.05])]:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=white)
        p.createMultiBody(0, 0, vis, pos, p.getQuaternionFromEuler(eul))
    body = p.loadURDF(urdf, [0, 0, 0], useFixedBase=True)
    red = [0.85, 0.10, 0.10, 1.0]
    for j in range(p.getNumJoints(body)):
        p.changeVisualShape(body, j, rgbaColor=red)
    p.changeVisualShape(body, -1, rgbaColor=red)
    view = p.computeViewMatrix([1.15, -1.15, 0.95], [0.35, 0.0, 0.35], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(55, 1.0, 0.05, 4.0)

    print("== GPU 上下文 ==")
    gr = GpuRenderer(body, urdf, joint_ids)
    print(f"  GL_RENDERER = {gr.info['GL_RENDERER']}")
    print(f"  GL_VERSION  = {gr.info['GL_VERSION']}")
    print(f"  GL_VENDOR   = {gr.info['GL_VENDOR']}")

    def bench(fn):
        out = {}
        for size in SIZES:
            for q in poses[:N_WARM]:
                fn(q, size)
            t0 = time.time()
            for q in poses[:N_TIME]:
                fn(q, size)
            out[size] = (time.time() - t0) / N_TIME * 1000.0
        return out

    print(f"\n== 计时（预热 {N_WARM} + 计时 {N_TIME} 帧，姿态=q_true[::15] 确定性） ==")
    tiny_ms = bench(lambda q, s: tiny_shoot(q, s, body, joint_ids, view, proj))
    gpu_ms = bench(lambda q, s: gr.render(q, s, view, proj))
    for size in SIZES:
        print(f"  {size}x{size}: CPU软渲染 {tiny_ms[size]:.1f} ms/帧 → GPU {gpu_ms[size]:.1f} ms/帧 "
              f"（加速 {tiny_ms[size] / gpu_ms[size]:.1f}×，判据基线 {CPU_MS[size]}ms）")

    print("\n== 同姿态画面对照（q_true[15]，128²） ==")
    q = poses[15 % len(poses)]
    im_t = tiny_shoot(q, 128, body, joint_ids, view, proj)
    im_g = gr.render(q, 128, view, proj)
    nt, ct = red_stats(im_t)
    ng, cg = red_stats(im_g)
    ee = project_ee(p.getLinkState(body, meta["ee_link"])[4], view, proj, 128)
    dist = float(np.hypot(ct[0] - cg[0], ct[1] - cg[1]))
    print(f"  Tiny: 红像素={nt} 红质心=({ct[0]:.1f},{ct[1]:.1f})")
    print(f"  GPU : 红像素={ng} 红质心=({cg[0]:.1f},{cg[1]:.1f})")
    print(f"  真值末端投影≈({ee[0]:.1f},{ee[1]:.1f})  质心相距={dist:.1f}px")

    gl_name = gr.info["GL_RENDERER"]
    ok1 = any(k in gl_name for k in ("NVIDIA", "GeForce", "Intel", "AMD", "Radeon", "Apple"))
    ok2 = ng > 0 and dist < 0.15 * 128
    ok3 = gpu_ms[128] < CPU_MS[128] and gpu_ms[224] < CPU_MS[224]
    print("\n== 判据裁定（开跑前定死） ==")
    print(f"  1) 真显卡光栅（{gl_name}）→ {'通过' if ok1 else '不通过'}")
    print(f"  2) 红臂可见且质心对齐（差 {dist:.1f}px < 19.2px）→ {'通过' if ok2 else '不通过'}")
    print(f"  3) 快于 CPU 基线（{gpu_ms[128]:.1f}/{gpu_ms[224]:.1f} vs {CPU_MS[128]}/{CPU_MS[224]}）→ {'通过' if ok3 else '不通过'}")
    print(f"  150 粒子/帧渲染预估：CPU {CPU_MS[128] * 150 / 1000:.2f}s → GPU {gpu_ms[128] * 150 / 1000:.2f}s（128²）")
    p.disconnect()


if __name__ == "__main__":
    main()
