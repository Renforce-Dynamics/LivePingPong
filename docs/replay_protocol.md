# LivePingPong Replay Protocol

`live_pingpong.replay.v1` is the upload contract for LivePingPong replay
visualization. It is intentionally a display-layer format: real-robot logs,
sim2sim logs, and converted mocap captures should normalize into this shape
before they are uploaded to the web demo.

The default replay shipped with the app is an example of this protocol,
converted from a PingPongFSM recorder `.npz`.

## Goals

- Make real-robot and sim2sim captures replayable through the same web UI.
- Keep the canonical fields independent of a specific recorder backend.
- Preserve enough planner, ball, racket, and robot state for debugging.
- Allow producers to add optional diagnostic fields without breaking older
  viewers.

## Containers

The preferred upload container is `.npz` produced by `numpy.savez` or
`numpy.savez_compressed`. The app also accepts converted `.json` files using
the same field names.

For browser-facing `.npz` files, use numeric and boolean arrays only for
per-tick data. Metadata should be written as UTF-8 JSON text in `meta_json`
when possible. Avoid pickle-backed object arrays for new producers; Python can
read them, but browser parsers intentionally skip pickle data.

Current PingPongFSM sources that should normalize to this protocol include:

```text
PingPongFSM/logs/recordings/landing_*.npz
PingPongFSM/logs/recordings/landing_mj_*.npz
PingPongFSM/logs/recordings/nokov_ball_*.npz
PingPongFSM/logs/recordings/ball_drop_*.npz
PingPongFSM/outputs/pingpongfsm_play_*.npz
```

Nokov L0 JSONL recordings should not be uploaded directly. Convert them first
with the existing perception/filter pipeline so the web viewer receives
world-frame ball, base, and racket data instead of raw mocap coordinates.

## Coordinate System

All canonical vectors use SI units and the MuJoCo/PingPongFSM world frame:

- X: forward, across the table toward the opponent.
- Y: left.
- Z: up.
- Position unit: metres.
- Velocity unit: metres per second.
- Angle unit: radians.
- Time unit: seconds.
- Quaternions are `[w, x, y, z]`.

Field names ending in `_w` are world-frame values. `rel_racket_target_pos_w`
is root-relative in position, but still expressed on world axes.

## Required Fields

These arrays are required for a full robot-and-ball replay:

| Field | Shape | Dtype | Meaning |
|---|---:|---|---|
| `t` | `(T,)` | float32/float64 | Seconds since recording start. Must be monotonic. |
| `ball_pos` | `(T, 3)` | float32/float64 | Ball position in world metres. |
| `ball_vel` | `(T, 3)` | float32/float64 | Ball velocity in world m/s. |
| `base_pos` | `(T, 3)` | float32/float64 | Policy-root base position used by the planner/policy/viewer. |
| `base_quat` | `(T, 4)` | float32/float64 | Policy/control base quaternion in `wxyz` order. |
| `qj` | `(T, J)` | float32/float64 | Robot joint positions in radians. |

For the current G1 web scene, `J` is expected to be 29 unless
`meta_json.joint_names` declares a different order that the viewer supports.

## Recommended Fields

These fields are not required to render the robot, but they make replay useful
for debugging policy and planner behavior:

| Field | Shape | Dtype | Meaning |
|---|---:|---|---|
| `dqj` | `(T, J)` | float | Robot joint velocities. |
| `qj_cmd` | `(T, J)` | float | PD target or command sent to the motors. |
| `base_lin_vel` | `(T, 3)` | float | Base linear velocity. |
| `ang_vel` | `(T, 3)` | float | Base angular velocity. |
| `ball_valid` | `(T,)` | bool | Whether `ball_pos` should be treated as a valid observation. |
| `fsm_state` | `(T,)` | int | FSM state code. |
| `supervisor_downgraded` | `(T,)` | bool | Whether supervisor downgraded the policy this tick. |
| `planner_status` | `(T,)` | int | Planner status code. |
| `planner_valid` | `(T,)` | bool | Whether planner output is valid. |
| `planner_reason_code` | `(T,)` | int | Encoded planner reason; decode through metadata. |
| `hit_pos_w` | `(T, 3)` | float | Predicted hit point, NaN when invalid. |
| `hit_vel_w` | `(T, 3)` | float | Predicted ball velocity at hit, NaN when invalid. |
| `target_landing_pos_w` | `(T, 3)` | float | Planned ball landing point, NaN when invalid. |
| `desired_ball_dir_w` | `(T, 3)` | float | Planned outgoing ball direction, NaN when invalid. |
| `time_to_hit_s` | `(T,)` | float | Time to hit, NaN when invalid. |
| `base_pos_target` | `(T, 2)` | float | Planner base XY target, NaN when invalid. |
| `rel_racket_target_pos_w` | `(T, 3)` | float | Planner/policy racket target relative to base root. |
| `racket_target_vel_w` | `(T, 3)` | float | Planner/policy racket target velocity. |
| `desired_racket_normal_w` | `(T, 3)` | float | Planner expected racket normal. |
| `planner_is_forehand` | `(T,)` | int8 | `-1` unknown, `0` backhand, `1` forehand. |
| `racket_pos_w` | `(T, 3)` | float | Measured or computed racket/contact point. |
| `racket_quat_wxyz` | `(T, 4)` | float | Racket/contact frame quaternion. |
| `racket_lin_vel_w` | `(T, 3)` | float | Racket/contact linear velocity. |
| `racket_ang_vel_w` | `(T, 3)` | float | Racket/contact angular velocity. |
| `racket_normal_w` | `(T, 3)` | float | Measured or computed racket normal. |
| `racket_ball_contact` | `(T,)` | bool | Contact marker from sim/debug data. |

Use NaN, not zero, for invalid optional float values. That lets the viewer
distinguish "no signal this tick" from a real world origin or zero command.

## Real-Robot Extensions

Real-robot logs may include the following optional diagnostic fields:

| Field | Shape | Meaning |
|---|---:|---|
| `base_raw_pos_w` | `(T, 3)` | Raw base pose before policy-root offset. |
| `base_policy_root_pos_w` | `(T, 3)` | Explicit alias of `base_pos`. |
| `robot_imu_quat_wxyz` | `(T, 4)` | Onboard IMU quaternion before mocap override. |
| `ball_phase_code` | `(T,)` | Encoded ball perception phase. |
| `ball_raw_valid` | `(T,)` | Raw ball detector validity. |
| `ball_n_samples` | `(T,)` | Number of raw ball samples used this tick. |
| `ball_filter_reason_code` | `(T,)` | Encoded ball filter reason. |
| `racket_pose_valid` | `(T,)` | Whether the racket rigid body was visible. |
| `racket_pose_source_code` | `(T,)` | Encoded racket pose source. |
| `racket_has_server_vel` | `(T,)` | Whether server-provided racket velocity was available. |
| `racket_raw_pos_w` | `(T, 3)` | Raw rigid-body origin. |
| `racket_raw_quat_wxyz` | `(T, 4)` | Raw rigid-body quaternion. |
| `racket_n_markers_visible` | `(T,)` | Visible racket marker count. |
| `racket_n_markers_total` | `(T,)` | Total configured racket marker count. |
| `racket_timestamp_s` | `(T,)` | Mocap wall-clock timestamp. |
| `racket_age_s` | `(T,)` | Controller time minus mocap timestamp. |

Unknown fields should be ignored by consumers and preserved by converters when
reasonable.

## Joint Order

The default `qj`, `dqj`, and `qj_cmd` order is the 29-DoF G1 MuJoCo order:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

If a producer writes a different joint order, it must set
`meta_json.joint_names`. A viewer may refuse unsupported orders instead of
guessing.

## Metadata

`meta_json` should be a JSON object with at least:

```json
{
  "protocol": "live_pingpong.replay.v1",
  "source": "real",
  "robot": "unitree_g1",
  "scene": "g1_scene_table_tennis_movable.xml",
  "dt": 0.02,
  "num_joints": 29,
  "joint_names": [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
  ],
  "base_pos_semantics": "policy_root_world",
  "quat_order": "wxyz",
  "units": {
    "time": "s",
    "position": "m",
    "velocity": "m/s",
    "joint_position": "rad",
    "joint_velocity": "rad/s"
  },
  "table": {
    "z_surface": 0.76,
    "x_min": 0.63,
    "x_max": 3.37,
    "y_min": -0.7625,
    "y_max": 0.7625,
    "net_x": 2.0,
    "ball_radius": 0.02
  },
  "reason_codes": {
    "0": "ok"
  },
  "ball_phase_codes": {},
  "ball_filter_reason_codes": {},
  "racket_pose_source_codes": {}
}
```

`source` should be one of `real`, `sim2sim`, `converted`, or a more specific
producer string. Keep absolute local paths out of shared files unless the file
is private to the lab machine.

## Relation to Ball-Only Captures

PingPongFSM also has a ball-only canonical trajectory format,
`pingpong.ball.v1`, used for mocap extraction and calibration. That format is
still the right L1 shape for "I recorded only the ball".

To replay in LivePingPong, convert ball-only captures into
`live_pingpong.replay.v1` by adding robot/base fields from the matching robot
log, or by generating placeholder robot fields if the viewer only needs a
ball-only mode. The current LivePingPong upload path expects the full required
field set above.

If a producer intentionally emits ball-only replay data, set
`meta_json.profile = "ball_only"` and include at minimum:

```text
t
ball_pos
ball_vel
ball_valid
```

The viewer can then render only the table, ball path, and event annotations.
This profile is a protocol-level allowance; the current app still requires the
full field set for uploaded `.npz` files.

## Conversion Notes

For PingPongFSM recorder files, prefer loading through
`pingpong_fsm.real.recording.load_trajectory()` in Python. It already handles
older `.npz` files and fills missing optional fields with NaN, `false`, or `-1`
as appropriate.

Recommended mappings:

| Protocol field | Preferred source |
|---|---|
| `base_pos` | `base_policy_root_pos_w`, falling back to `base_pos` |
| `base_raw_pos_w` | Raw mocap/base pose before policy-root offset |
| `ball_pos` | Filtered world-frame ball position |
| `ball_valid` | Finite `ball_pos` plus producer validity signal when present |
| `racket_pos_w` | Measured racket/contact debug point |
| `racket_ball_contact` | MuJoCo contact in sim; optional inferred event in real logs |

Do not make the browser responsible for mocap coordinate transforms, marker
filtering, or base policy-root offsets. Those belong in the conversion step so
the upload file is self-contained and replayable offline.

## Validation Rules

A producer should validate before publishing an upload file:

- Every per-tick field has first dimension `T == len(t)`.
- `t` is finite and monotonic.
- Required arrays contain finite values.
- Optional invalid planner/racket fields use NaN.
- `base_quat` and `racket_quat_wxyz`, when valid, are normalized.
- `qj.shape[1]` matches `meta_json.num_joints`.
- Joint order is either the default order or declared in `meta_json.joint_names`.
- Dtypes are numeric or boolean; no pickle/object payload is needed for browser
  replay.

## Current Producers

PingPongFSM `TrajectoryRecorder` currently writes a compatible `.npz` with
`recorder_version` 5. Existing files may store `meta_json` as a pickle-backed
object scalar; Python loaders can read that, and the web uploader will still
use the numeric arrays. New shared files should store metadata as plain JSON
text.

Older sim2sim examples may be missing newer real-robot diagnostics such as
`base_raw_pos_w`, `robot_imu_quat_wxyz`, full racket pose, and extra planner
vectors. Converters should preserve the shared core fields and fill missing
diagnostics rather than rejecting the file.
