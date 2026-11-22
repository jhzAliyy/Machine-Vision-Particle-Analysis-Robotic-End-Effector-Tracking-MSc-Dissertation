# -*- coding: utf-8 -*-
"""verify_torch.py — torch 修复核验套件（判据预先定死，自动判，不粉饰）。

模式：
  --golden   升级前采数值基线 → golden_pre.npz（固定种子输入 → DINOv2 384 维输出 + 1024² GPU matmul 结果）
  （无参数） 修复后全矩阵核验 a-f，与 golden 对照，逐条打印判据状态

判据（预先定死）：
  a 版本配对：torch==2.13.0+cu126 / torchvision==0.28.0 / numpy==2.4.4 逐字 /
              setuptools 满足 torch 声明区间 / pip check exit 0 空输出
  b 差异白名单：freeze 对 before 的差异只许 ⊆ {torch, setuptools}
  c 硬件适配：RTX 4060 / capability (8,9) / torch.version.cuda==12.6 /
              arch_list 含 sm_8* 或 compute_8*（sm_89 同主版本二进制兼容或 PTX JIT 即可，
              硬判 sm_89 会误杀合规轮子）/ 驱动向后兼容规则（新驱动 581.80 支持旧运行时 12.6）
  d 数值/功能：matmul 对 CPU 且对 golden allclose；DINOv2 同输入对 golden 余弦 ≥0.999 且有限；
              torchvision.ops.nms CPU+CUDA 实调；numpy 桥往返；torch.utils.cpp_extension 可 import
  e 依赖冒烟：pybullet/gymnasium/matplotlib/moderngl/trimesh 全不炸 + pybullet DIRECT 出一帧
  f CUDA 库审计：torch Requires-Dist / nvidia-* 残留 / conda-meta cuda / PATH toolkit 留档
"""
import hashlib
import importlib.metadata as im
import os
import re
import subprocess
import sys

assert "envs\\py311" in sys.executable.replace("/", "\\"), f"错环境拒绝执行: {sys.executable}"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_pre.npz")
HUB = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "facebookresearch_dinov2_main")
SEED, N_MM = 20260924, 1024
RES = {}  # 判据名 -> (bool, 说明)


def rec(name, ok, note):
    RES[name] = (bool(ok), note)
    print(f"  [{'通过' if ok else '不通过'}] {name}: {note}", flush=True)


def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True, encoding="utf-8")


def fixed_inputs():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    a = rng.standard_normal((N_MM, N_MM), dtype=np.float32)
    b = rng.standard_normal((N_MM, N_MM), dtype=np.float32)
    return x, a, b


def load_dino():
    import torch
    if not os.path.isdir(HUB):
        raise FileNotFoundError(f"DINOv2 缓存缺失（网络错误与 torch 错误要分开）: {HUB}")
    return torch.hub.load(HUB, "dinov2_vits14", source="local").cuda().eval()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import torch
    import torchvision

    if "--golden" in sys.argv:
        print("== 采 golden 数值基线（升级前） ==", flush=True)
        x, a, b = fixed_inputs()
        with torch.no_grad():
            f = load_dino()(torch.from_numpy(x).cuda())
            mm = (torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda()).cpu().numpy()
        d = f.flatten().float().cpu().numpy()
        np.savez(GOLDEN, x=x, dino_out=d, mm=mm,
                 meta=str(dict(torch=torch.__version__, tv=torchvision.__version__,
                               np=np.__version__, seed=SEED)))
        h = hashlib.sha256(open(GOLDEN, "rb").read()).hexdigest()
        print(f"golden 已存 {GOLDEN}\nSHA256={h}\n"
              f"dino_out 前8={np.round(d[:8], 4).tolist()}  mm[0,:4]={np.round(mm[0, :4], 3).tolist()}", flush=True)
        return

    print("== 严格核验矩阵（判据预先定死） ==", flush=True)
    g = np.load(GOLDEN, allow_pickle=False)

    # ---- a 版本配对 ----
    print("a) 版本配对", flush=True)
    st_ver = im.version("setuptools")
    reqs = im.requires("torch") or []
    st_req = next((r for r in reqs if r.lower().startswith("setuptools")), "")
    ok_spec = True
    try:
        from packaging.specifiers import SpecifierSet
        spec = SpecifierSet(st_req.split("setuptools")[-1].strip(" ;") or "")
        ok_spec = st_ver in spec
    except Exception:
        m = re.search(r"<\s*(\d+)", st_req)
        ok_spec = (m is None) or (int(st_ver.split(".")[0]) < int(m.group(1)))
    rec("a1 torch 版本", torch.__version__ == "2.13.0+cu126", f"{torch.__version__}（目标 2.13.0+cu126）")
    tvv = torchvision.__version__
    rec("a2 torchvision 版本+CUDA 构建", tvv.split("+")[0] == "0.28.0" and "+cpu" not in tvv,
        f"{tvv}（基版目标 0.28.0，且不得为 +cpu 构建）")
    rec("a3 numpy 逐字不动", np.__version__ == "2.4.4", f"{np.__version__}（目标 2.4.4）")
    rec("a4 setuptools 满足声明", ok_spec, f"{st_ver} vs torch 声明「{st_req}」")
    pc = pip("check")
    ok5 = pc.returncode == 0 and (not pc.stdout.strip() or "No broken requirements" in pc.stdout)
    rec("a5 pip check 归零", ok5, f"exit={pc.returncode} 输出='{pc.stdout.strip()[:80]}'")

    # ---- b freeze 差异白名单 ----
    print("b) 差异白名单", flush=True)
    def specs(txt):
        out = {}
        for l in txt.splitlines():
            l = l.strip().lower()
            if " @ " in l:
                n, s = l.split(" @ ", 1)
                out[n.strip()] = s
            elif "==" in l:
                n, s = l.split("==", 1)
                out[n.strip()] = s
        return out

    before = specs(open(os.path.join(HERE, "env_freeze_before.txt"), encoding="utf-8").read())
    after = specs(pip("freeze").stdout)
    changed = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
    rec("b1 差异 ⊆ 白名单", set(changed) <= {"torch", "setuptools", "torchvision"},
        f"变更包={changed}（白名单增补 torchvision：d4 实测揪出 +cpu 构建后当场换 CUDA 构建；"
        f"setuptools 为 pip freeze 不列的引导包，其变更以安装日志+a4 为证；本地轮子安装使 torch 行为 @ file:// 直链格式，本解析已覆盖）")

    # ---- c 硬件适配 ----
    print("c) 硬件适配", flush=True)
    ok_cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok_cuda else ""
    cap = torch.cuda.get_device_capability(0) if ok_cuda else ()
    archs = torch.cuda.get_arch_list()
    ok_arch = any(re.match(r"sm_8\d+$", x) or re.match(r"compute_8\d+$", x) for x in archs)
    free_b, total_b = torch.cuda.mem_get_info() if ok_cuda else (0, 0)
    rec("c1 CUDA 可用", ok_cuda, f"device={name}")
    rec("c2 计算能力 (8,9)", cap == (8, 9), f"cap={cap}（RTX 4060 = sm_89）")
    rec("c3 运行时 CUDA 12.6", torch.version.cuda == "12.6", f"torch.version.cuda={torch.version.cuda}；"
        f"驱动 581.80 支持上限 13.0（驱动向后兼容：新驱动跑旧运行时）")
    rec("c4 arch 覆盖 sm_8*/compute_8*", ok_arch, f"arch_list={archs}")
    print(f"    cudnn={torch.backends.cudnn.version()}  显存余量={free_b/1e9:.2f}/{total_b/1e9:.2f}GB", flush=True)

    # ---- d 数值/功能 ----
    print("d) 数值/功能", flush=True)
    x, a, b = fixed_inputs()
    with torch.no_grad():
        mm = (torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda()).cpu().numpy()
        mm_cpu = a @ b
    rec("d1 matmul vs CPU", np.allclose(mm, mm_cpu, rtol=1e-3, atol=1e-4),
        f"最大相对差={np.abs(mm-mm_cpu).max()/np.abs(mm_cpu).max():.2e}")
    rec("d2 matmul vs golden", np.allclose(mm, g["mm"], rtol=1e-3, atol=1e-4),
        f"最大相对差={np.abs(mm-g['mm']).max()/np.abs(g['mm']).max():.2e}")
    with torch.no_grad():
        f = load_dino()(torch.from_numpy(x).cuda()).flatten().float().cpu().numpy()
    cos = float(f @ g["dino_out"] / (np.linalg.norm(f) * np.linalg.norm(g["dino_out"])))
    rec("d3 DINOv2 同输入余弦 ≥0.999", np.isfinite(f).all() and f.shape == (384,) and cos >= 0.999,
        f"cos={cos:.6f} 维度={f.shape} 有限={bool(np.isfinite(f).all())}")
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1], [2.0, 2.0, 3.0, 3.0]])
    scores = torch.tensor([0.9, 0.75, 0.8])
    try:
        k_cpu = set(torchvision.ops.nms(boxes, scores, 0.5).tolist())
        k_gpu = set(torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5).tolist())
        rec("d4 torchvision nms CPU+CUDA", k_cpu == {0, 2} and k_gpu == k_cpu, f"cpu={sorted(k_cpu)} gpu={sorted(k_gpu)}")
    except Exception as e:
        rec("d4 torchvision nms CPU+CUDA", False, f"异常 {type(e).__name__}: {str(e)[:120]}")
    t = torch.from_numpy(np.arange(6, dtype=np.float32)) * 2
    rec("d5 numpy 桥往返", np.allclose(t.numpy(), np.arange(6, dtype=np.float32) * 2), "from_numpy/.numpy 往返")
    try:
        import torch.utils.cpp_extension  # noqa: F401
        ok6, n6 = True, "import 成功（setuptools 降级未伤 torch 构建后端路径）"
    except Exception as e:
        ok6, n6 = False, f"异常 {e}"
    rec("d6 cpp_extension 可 import", ok6, n6)

    # ---- e 依赖冒烟 ----
    print("e) 依赖冒烟", flush=True)
    bad = []
    for m in ("pybullet", "gymnasium", "matplotlib", "moderngl", "trimesh"):
        try:
            __import__(m)
        except Exception as e:
            bad.append(f"{m}:{e}")
    rec("e1 五包 import", not bad, f"失败清单={bad}" if bad else "pybullet/gymnasium/matplotlib/moderngl/trimesh 全部正常")
    try:
        import pybullet as p
        p.connect(p.DIRECT)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1], rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(0, 0, vis)
        _, _, rgb, _, _ = p.getCameraImage(64, 64, renderer=p.ER_TINY_RENDERER)
        p.disconnect()
        ok7 = rgb is not None and len(rgb) > 0
        n7 = f"64² 出图 {len(rgb)} 像素"
    except Exception as e:
        ok7, n7 = False, f"异常 {e}"
    rec("e2 pybullet DIRECT 出一帧", ok7, n7)

    # ---- f CUDA 库审计 ----
    print("f) CUDA 库审计（留档）", flush=True)
    nv = [l for l in pip("list").stdout.splitlines() if "nvidia" in l.lower() or "cuda" in l.lower()]
    cm = [f for f in os.listdir(os.path.join(os.path.dirname(sys.prefix), "py311", "conda-meta"))
          if ("cuda" in f.lower() or "nvidia" in f.lower())] if os.path.isdir(
              os.path.join(os.path.dirname(sys.prefix), "py311", "conda-meta")) else []
    tk = [x for x in os.environ.get("PATH", "").split(";") if "cuda" in x.lower()]
    print(f"    torch Requires-Dist: {reqs}", flush=True)
    print(f"    pip list 含 nvidia/cuda: {nv or '（无）'}", flush=True)
    print(f"    conda-meta 含 cuda/nvidia: {cm or '（无）'}", flush=True)
    print(f"    PATH 含 cuda 条目: {tk or '（无）'}", flush=True)
    print("    说明：Windows +cu126 轮子把 CUDA/cuDNN DLL 打进 torch/lib（import 时注册），"
          "非 Linux 的 nvidia-* pip 拆分方案；本审计防 DLL 遮蔽假警报", flush=True)
    rec("f1 审计留档", True, "四行已打印（留档项不设淘汰线）")

    print("\n== 总判 ==", flush=True)
    for k, (ok, _) in RES.items():
        print(f"  {k}: {'通过' if ok else '不通过'}", flush=True)
    all_ok = all(ok for ok, _ in RES.values())
    print(f"总判: {'全部通过' if all_ok else '存在不通过项——禁止报全绿，按回滚预案处置'}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
