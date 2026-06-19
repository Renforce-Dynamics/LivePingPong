# PingPongFSM Web Demo

Browser replay demo for the PingPongFSM sim2sim workflow.

The app has two modes:

- `Replay`: recorded MuJoCo playback. It loads the G1 table-tennis MJCF scene in MuJoCo WASM, applies the recorded `qpos` trajectory, and overlays the ball trail, racket position, planner hit point, base target, and FSM/planner state.
- `Physics`: browser-side sim2sim-lite. It samples incoming balls, runs MuJoCo `mj_step()`, builds the HRLHit-MJ 104-D observation, runs `policy_beta.onnx` through `onnxruntime-web`, applies PD torque, and uses a browser-side one-way impact fallback to keep returns robust when MuJoCo JS contact details are unavailable.

Replay defaults to `public/trajectories/pingpongfsm_play_20s.json`. Use `Upload NPZ` in the UI to load recorder `.npz` files directly in the browser; converted `.json` files are also accepted.

## Run

```bash
npm install
npm run dev -- --host 0.0.0.0 --port 5175
```

Production check:

```bash
npm run build
npm run preview -- --host 0.0.0.0 --port 4175
```

## Source Data

- Scene: `public/assets/g1_description/g1_scene_table_tennis_movable.xml`
- Robot: `public/assets/g1_description/g1_robot_movable_base.xml`
- Meshes: `public/assets/g1_description/meshes/`
- Trajectory: `public/trajectories/pingpongfsm_play_20s.json`
- Policy: `public/policy_beta.onnx`

The trajectory was converted from:

`/data/user_data/users/metabot-workspace/pingpong/outputs/pingpongfsm_play_20s.npz`

## Next Step

The remaining gap to the Python sim2sim runner is the full landing planner/supervisor/FSM parity. Physics mode currently implements the policy observation, ONNX policy, PD loop, MuJoCo stepping, sampled serves, and a simplified hit-plane planner.
