# LivePingPong Replay Upload

LivePingPong accepts replay files that use the `live_pingpong.replay.v1`
field names. The public demo treats this as a visualization format, not as a
production robot log.

## Containers

- `.npz`: preferred for normal uploads because frame arrays stay compact.
- `.json`: accepted for small demos and debugging.

Metadata should be plain JSON text. Do not require pickle/object payloads for
browser replay.

## Required Arrays

The current replay viewer expects these arrays:

| Field | Shape | Meaning |
|---|---:|---|
| `t` | `(T,)` | Time in seconds. |
| `ball_pos` | `(T, 3)` | Ball position in world metres. |
| `ball_vel` | `(T, 3)` | Ball velocity in world m/s. |
| `base_pos` | `(T, 3)` | Robot base position used for visualization. |
| `base_quat` | `(T, 4)` | Robot base quaternion in `wxyz` order. |
| `qj` | `(T, J)` | Robot joint positions in radians. |

Optional arrays such as joint velocity, planner overlays, racket overlays, and
contact markers may be included. Missing optional arrays are ignored or filled
with neutral defaults by the viewer.

## Policy Files

The public repository does not include a policy model. To use Physics mode,
upload an ONNX policy from the UI. Replay mode works without a policy file.

## Compatibility Rules

- Use SI units: seconds, metres, metres per second, radians.
- Use `wxyz` quaternion order.
- Keep all per-frame arrays aligned to the same `T`.
- Use NaN for invalid optional float overlays.
- Keep browser upload files free of local paths, private config names, and raw
  capture metadata.

Converters may preserve richer private diagnostics in internal archives, but
shared replay files should only include what the visualization needs.
