# PingPongFSM Web Demo

Browser replay demo for the PingPongFSM sim2sim workflow.

The first version is intentionally a recorded MuJoCo playback: it loads the G1 table-tennis MJCF scene in MuJoCo WASM, applies the recorded `qpos` trajectory, and overlays the ball trail, racket position, planner hit point, base target, and FSM/planner state. This keeps the web demo stable while leaving the live ONNX/planner/FSM port as a later step.

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

The trajectory was converted from:

`/data/user_data/users/metabot-workspace/pingpong/outputs/pingpongfsm_play_20s.npz`

## Next Step

The live browser closed loop would need a TypeScript port of the landing planner, FSM state transitions, policy observation builder, and PD control loop, plus `onnxruntime-web` for the ONNX policy. The current replay mode is the safe first demo surface for sharing and visualization.
