# _mvp 复现说明（最小可行验证）

## 依赖
- Python 环境（torch 2.13.0+cu126 用于 DINOv2 打分；torch/torchvision 一律走 cu126 索引钉 `==x.y.z+cu126`，禁止 +cpu 构建，见 workspace/pyenv.md；pybullet / moderngl / trimesh / pycollada / rosbags 为已装依赖）
- 网格文件：ros-industrial/universal_robot 的 `ur_description/meshes/ur5/`（visual/*.dae + collision/*.stl），
  放到 `_mvp/vendor/**/ur_description/meshes/ur5/`。本目录自带的 `ur5.urdf` 为从官方 xacro
  （universal_robot@1.2.5）展开的纯 URDF 基线，运动学数值与官方一致。

## 运行
```
python gen_data.py    # 造 300 帧 UR5 序列（自渲染观测，含关节/末端真值）
python run_pf.py      # 五组实验：死推算 + 粒子滤波 2×2（像素比对/DINOv2 打分 × 带提示/随机走）
```
产出：`results.md`（结果表+判据裁定）、`fig1_误差曲线.png`、`fig2_先验价值.png`。

## 路径
本目录路径已经全英文（`…/uk_supervisor/dissertation/_mvp`），pybullet 可直接读取，无需任何盘符映射。
若把目录移回含中文的路径并报 "URDF not found"，请改用英文路径——不要用虚拟盘。

## 设计要点（对应方案判据）
- 观测=自渲染（仿真内零域差），只证机制，不背真实世界承诺
- 跟踪侧预埋 ±2°/关节恒定偏差（模拟电机读数不准），死推算基线中位误差约 24mm
- 粒子状态=6 关节角；带提示组 transition=关节增量+0.8° 噪声；随机走组=3.5° 扩散（DiffPF 式"动作不可得"）
- 判据：带提示 PF 中位误差 ≤10mm 且比死推算低 ≥50%；比随机走低 ≥30%；DINOv2 组显存 <6GB、每帧 ≤3s

## 2026-09-24 阶段增补工具
- `gpu_probe.py`：pybullet 渲染路径取证（结论：无真 GPU 路径，EGL 缺失/硬 OpenGL 静默回退）
- `gpu_render.py`：真 GPU 渲染器（moderngl 光栅 + pybullet 运动学），三判据全过（0.5ms/帧@128）
- `run_pf_gpu.py`：GPU 渲染版五组实验（与 run_pf.py 逐位同源，唯一变量=渲染器）→ `results_gpu.md`
- `verify_torch.py`：环境核验矩阵（`--golden` 采数值基线；无参数=19 项全矩阵判定）
- `SEEDS.md`：随机种子序列存档与复核记录（smoke 逐点一致）
- `bag_info.py` / `bag_tf_frames.py`：Apollo ROS bag 摸底（话题/类型/坐标系对）
- `apollo_extract.py`：Apollo 素材抽取+真值组装（头标标定双观测一致性 0.1mm）
- `apollo_probe.py`：真照片判别力抽查（凸包 IoU + 深度残差双打分，扰动分轴判定）
- 运行在本目录直接执行；Apollo 原始包在 `D:\apollo_dataset\`、
  工作产物（apollo_extract.npz 等）在 `D:\apollo_work\`；官方基准/论文口径见阶段报告 report_0924.pdf
