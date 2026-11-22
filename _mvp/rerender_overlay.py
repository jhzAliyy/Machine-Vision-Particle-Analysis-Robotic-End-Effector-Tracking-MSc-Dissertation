# -*- coding: utf-8 -*-
"""rerender_overlay.py — 从 apollo_overlay.npz 秒级重渲染（不重跑跟踪）。样式改动后出片用。"""
import os as _os
_AW = _os.environ.get("APOLLO_WORK", "D:/apollo_work")
_AD = _os.environ.get("APOLLO_DATASET", "D:/apollo_dataset")
import json
import os
import shutil
import sys

import numpy as np

from apollo_overlay import FRAME_DIR, OUT_MP4, STILL_DIR, encode_mp4, render

NPZ = _os.path.join(_AW, "apollo_overlay.npz")
CONST_NPZ = _os.path.join(_AW, "apollo_extract.npz")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    z = np.load(NPZ, allow_pickle=False)
    K = z["K"]
    cst = np.load(CONST_NPZ, allow_pickle=False)
    pf = json.loads(str(cst["palm_frames"]))
    T_a2p = np.array(pf["/R_ARM_PALM|/R_PALM"])
    hp = np.load(_os.path.join(_AW, "hand_points.npy"))
    hp4 = np.concatenate([hp, np.ones((len(hp), 1))], 1) @ np.linalg.inv(T_a2p).T
    hp = hp4[:, :3].astype(np.float32)
    if os.path.isdir(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    render(list(z["rgb"]), K, hp, z["est"], z["base"], z["gt"], z["errs"], z["errs0"], z["bhist"], FRAME_DIR)
    encode_mp4(FRAME_DIR, OUT_MP4)
    shutil.copy(OUT_MP4, os.path.join(STILL_DIR, "apollo_overlay.mp4"))
    for k in (9, 49, 89):
        shutil.copy(f"{FRAME_DIR}/frame_{k:03d}.png", os.path.join(STILL_DIR, f"overlay_still_f{k+1:03d}.png"))
    print("[done] 重渲染完成")


if __name__ == "__main__":
    main()
