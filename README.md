# PingPongFSM Web Demo

Browser replay demo for a table-tennis robot workflow.

The app has two modes:

- `Replay`: recorded MuJoCo playback with ball, racket, target, and contact overlays.
- `Physics`: browser-side simplified physics mode. It runs MuJoCo WASM and a user-uploaded ONNX policy through `onnxruntime-web`.

Replay defaults to `public/trajectories/pingpongfsm_play_20s.json`. Use `Upload NPZ` in the UI to load replay `.npz` files directly in the browser; converted `.json` files are also accepted. Use `Upload ONNX` to enable Physics mode. The upload contract is documented in [`docs/replay_protocol.md`](docs/replay_protocol.md).

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

## Convert Replay Files

```bash
python3 scripts/convert_replay_npz.py --out-dir ./converted input.npz
```

## Source Data

- Scene: `public/assets/g1_description/g1_scene_table_tennis_movable.xml`
- Robot: `public/assets/g1_description/g1_robot_movable_base.xml`
- Meshes: `public/assets/g1_description/meshes/`
- Trajectory: `public/trajectories/pingpongfsm_play_20s.json`
Policy files are intentionally not committed. Upload an ONNX file in the UI to
run Physics mode.

## Next Step

Physics mode is a browser demo. It is not the production controller.
