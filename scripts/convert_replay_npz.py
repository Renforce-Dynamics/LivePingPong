#!/usr/bin/env python3
"""Convert recorder NPZ files into browser-safe LivePingPong replay NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "live_pingpong.replay.v1"
REQUIRED_FIELDS = ("t", "ball_pos", "ball_vel", "base_pos", "base_quat", "qj")
PREFERRED_FIELDS = (
    "t",
    "ball_pos",
    "ball_vel",
    "base_pos",
    "base_raw_pos_w",
    "base_policy_root_pos_w",
    "base_quat",
    "robot_imu_quat_wxyz",
    "base_lin_vel",
    "ang_vel",
    "qj",
    "dqj",
    "qj_cmd",
    "fsm_state",
    "supervisor_downgraded",
    "planner_status",
    "hit_pos_w",
    "hit_vel_w",
    "target_landing_pos_w",
    "desired_ball_dir_w",
    "time_to_hit_s",
    "base_pos_target",
    "planner_valid",
    "planner_reason_code",
    "ball_phase_code",
    "ball_raw_valid",
    "ball_n_samples",
    "ball_filter_reason_code",
    "rel_racket_target_pos_w",
    "racket_target_vel_w",
    "desired_racket_normal_w",
    "planner_is_forehand",
    "racket_face_axis_local",
    "racket_pos_w",
    "racket_pose_valid",
    "racket_pose_source_code",
    "racket_quat_wxyz",
    "racket_lin_vel_w",
    "racket_ang_vel_w",
    "racket_has_server_vel",
    "racket_normal_w",
    "racket_raw_pos_w",
    "racket_raw_quat_wxyz",
    "racket_n_markers_visible",
    "racket_n_markers_total",
    "racket_timestamp_s",
    "racket_age_s",
    "racket_ball_contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="input recorder .npz files")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for converted files")
    parser.add_argument(
        "--source",
        default="converted",
        help="public source label to write into metadata, e.g. demo/real/sim2sim/converted",
    )
    return parser.parse_args()


def read_meta(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "meta_json" not in npz.files:
        return {}
    try:
        item = npz["meta_json"].item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str) and item:
            raw = json.loads(item)
            return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}
    return {}


def default_arrays(arrays: dict[str, np.ndarray], n: int, j: int) -> None:
    f32 = np.float32
    nan = f32(np.nan)
    defaults: dict[str, np.ndarray] = {
        "ball_vel": np.zeros((n, 3), dtype=f32),
        "base_raw_pos_w": np.full((n, 3), nan, dtype=f32),
        "base_policy_root_pos_w": arrays.get("base_pos", np.zeros((n, 3), dtype=f32)),
        "robot_imu_quat_wxyz": np.full((n, 4), nan, dtype=f32),
        "base_lin_vel": np.zeros((n, 3), dtype=f32),
        "ang_vel": np.zeros((n, 3), dtype=f32),
        "dqj": np.zeros((n, j), dtype=f32),
        "qj_cmd": arrays.get("qj", np.zeros((n, j), dtype=f32)),
        "fsm_state": np.full(n, 13, dtype=np.int8),
        "supervisor_downgraded": np.zeros(n, dtype=np.bool_),
        "planner_status": np.full(n, -1, dtype=np.int8),
        "hit_pos_w": np.full((n, 3), nan, dtype=f32),
        "hit_vel_w": np.full((n, 3), nan, dtype=f32),
        "target_landing_pos_w": np.full((n, 3), nan, dtype=f32),
        "desired_ball_dir_w": np.full((n, 3), nan, dtype=f32),
        "time_to_hit_s": np.full(n, nan, dtype=f32),
        "base_pos_target": np.full((n, 2), nan, dtype=f32),
        "planner_valid": np.zeros(n, dtype=np.bool_),
        "planner_reason_code": np.full(n, -1, dtype=np.int16),
        "ball_phase_code": np.full(n, -1, dtype=np.int16),
        "ball_raw_valid": np.zeros(n, dtype=np.bool_),
        "ball_n_samples": np.zeros(n, dtype=np.int16),
        "ball_filter_reason_code": np.full(n, -1, dtype=np.int16),
        "rel_racket_target_pos_w": np.full((n, 3), nan, dtype=f32),
        "racket_target_vel_w": np.full((n, 3), nan, dtype=f32),
        "desired_racket_normal_w": np.full((n, 3), nan, dtype=f32),
        "planner_is_forehand": np.full(n, -1, dtype=np.int8),
        "racket_face_axis_local": np.full((n, 3), nan, dtype=f32),
        "racket_pos_w": np.full((n, 3), nan, dtype=f32),
        "racket_pose_valid": np.zeros(n, dtype=np.bool_),
        "racket_pose_source_code": np.full(n, -1, dtype=np.int16),
        "racket_quat_wxyz": np.full((n, 4), nan, dtype=f32),
        "racket_lin_vel_w": np.full((n, 3), nan, dtype=f32),
        "racket_ang_vel_w": np.full((n, 3), nan, dtype=f32),
        "racket_has_server_vel": np.zeros(n, dtype=np.bool_),
        "racket_normal_w": np.full((n, 3), nan, dtype=f32),
        "racket_raw_pos_w": np.full((n, 3), nan, dtype=f32),
        "racket_raw_quat_wxyz": np.full((n, 4), nan, dtype=f32),
        "racket_n_markers_visible": np.zeros(n, dtype=np.int16),
        "racket_n_markers_total": np.zeros(n, dtype=np.int16),
        "racket_timestamp_s": np.full(n, np.nan, dtype=np.float64),
        "racket_age_s": np.full(n, nan, dtype=f32),
        "racket_ball_contact": np.zeros(n, dtype=np.bool_),
    }
    for key, value in defaults.items():
        arrays.setdefault(key, value)


def convert_one(src: Path, out_dir: Path, source_label: str) -> dict[str, Any]:
    npz = np.load(src, allow_pickle=True)
    missing = [field for field in REQUIRED_FIELDS if field not in npz.files]
    if missing:
        raise ValueError(f"{src} missing required fields: {', '.join(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for key in PREFERRED_FIELDS:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.dtype == object:
            continue
        arrays[key] = arr

    n = int(arrays["t"].shape[0])
    j = int(arrays["qj"].shape[1])
    default_arrays(arrays, n, j)

    if "base_policy_root_pos_w" in arrays:
        arrays["base_pos"] = np.asarray(arrays["base_policy_root_pos_w"], dtype=np.float32)

    ball_finite = np.isfinite(arrays["ball_pos"]).all(axis=1)
    raw_valid = arrays.get("ball_raw_valid")
    if raw_valid is not None and raw_valid.shape[0] == n and np.any(raw_valid):
        arrays["ball_valid"] = (ball_finite & raw_valid.astype(bool)).astype(np.bool_)
    else:
        arrays["ball_valid"] = ball_finite.astype(np.bool_)

    source_meta = read_meta(npz)
    meta = {
        "protocol": PROTOCOL,
        "source": source_label,
        "source_basename": src.name,
        "recorded_ticks": n,
        "recorded_duration_s": float(arrays["t"][-1]) if n else 0.0,
        "num_joints": j,
        "dt": float(source_meta.get("dt", 0.02)),
        "units": {
            "time": "s",
            "position": "m",
            "velocity": "m/s",
            "joint_position": "rad",
            "joint_velocity": "rad/s",
        },
    }
    meta_text = json.dumps(meta, ensure_ascii=True, sort_keys=True)
    meta_bytes = meta_text.encode("utf-8")
    arrays["meta_json"] = np.array(meta_bytes, dtype=f"S{len(meta_bytes)}")
    arrays["meta_json_utf8"] = np.frombuffer(meta_bytes, dtype=np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}.live_pingpong.replay.v1.npz"
    np.savez_compressed(out_path, **arrays)

    # Validate that browser consumers do not need pickle support.
    clean = np.load(out_path, allow_pickle=False)
    for field in REQUIRED_FIELDS:
        _ = clean[field]
    object_fields = [field for field in clean.files if clean[field].dtype == object]
    if object_fields:
        raise ValueError(f"{out_path} contains object arrays: {object_fields}")

    return {
        "file": out_path.name,
        "source_basename": src.name,
        "ticks": n,
        "duration_s": float(arrays["t"][-1]) if n else 0.0,
        "num_joints": j,
        "size_bytes": out_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    manifest = {"protocol": PROTOCOL, "generated_files": []}
    for src in args.inputs:
        info = convert_one(src, args.out_dir, args.source)
        manifest["generated_files"].append(info)
        print(f"wrote {info['file']} ({info['size_bytes'] / 1e6:.2f} MB)")
    manifest_path = args.out_dir / "live_pingpong_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
