import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';

import { loadMuJoCoScene, updateSceneTransforms, type MuJoCoScene } from './mujocoScene';
import { PhysicsController, type JointSlots, type PhysicsSample } from './physicsController';
import { duration, fsmName, loadTrajectory, sampleIndex, type PingPongTrajectory } from './trajectory';

const SCENE_XML_URL = './assets/g1_description/g1_scene_table_tennis_movable.xml';
const MESH_BASE_URL = './assets/g1_description/meshes/';
const TRAJECTORY_URL = './trajectories/pingpongfsm_play_20s.json';
const VFS_SCENE_PATH = '/working/g1_scene_table_tennis_movable.xml';

type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

interface UI {
  playBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  serveBtn: HTMLButtonElement;
  replayBtn: HTMLButtonElement;
  physicsBtn: HTMLButtonElement;
  speedInput: HTMLInputElement;
  speedText: HTMLSpanElement;
  timeInput: HTMLInputElement;
  timeText: HTMLSpanElement;
  durationText: HTMLSpanElement;
  fsmText: HTMLElement;
  plannerText: HTMLElement;
  hitText: HTMLElement;
  ballText: HTMLElement;
  baseText: HTMLElement;
  contactText: HTMLElement;
  physicsStatus: HTMLElement;
}

function toThree(v: number[], target = new THREE.Vector3()): THREE.Vector3 {
  return target.set(v[0] ?? 0, v[2] ?? 0, -(v[1] ?? 0));
}

function finiteVec(v?: number[] | null): v is number[] {
  return Array.isArray(v) && v.length >= 3 && v.every((x) => Number.isFinite(x));
}

async function setupVFS(mujoco: MujocoModule) {
  mkdirp(mujoco, '/working');
  mkdirp(mujoco, '/working/meshes');

  const parser = new DOMParser();
  const xmlSources: string[] = [];
  const loadedXml = new Set<string>();

  async function loadXml(url: string, vfsPath: string) {
    if (loadedXml.has(vfsPath)) return;
    loadedXml.add(vfsPath);

    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch XML: ${url} (${res.status} ${res.statusText})`);
    }
    const text = normalizeMuJoCoXml(await res.text());
    mujoco.FS.writeFile(vfsPath, text);
    xmlSources.push(text);

    const xmlDoc = parser.parseFromString(text, 'text/xml');
    const includes = Array.from(xmlDoc.querySelectorAll('include[file]'))
      .map((el) => el.getAttribute('file'))
      .filter(Boolean) as string[];
    await Promise.all(includes.map((file) => {
      const childUrl = new URL(file, url).toString();
      const childPath = `${dirname(vfsPath)}/${basename(file)}`;
      return loadXml(childUrl, childPath);
    }));
  }

  await loadXml(new URL(SCENE_XML_URL, window.location.href).toString(), VFS_SCENE_PATH);

  const meshFiles = Array.from(new Set(xmlSources.flatMap((xml) => {
    const xmlDoc = parser.parseFromString(xml, 'text/xml');
    return Array.from(xmlDoc.querySelectorAll('mesh[file]'))
      .map((el) => el.getAttribute('file'))
      .filter(Boolean) as string[];
  })));

  const missing: string[] = [];
  await Promise.all(meshFiles.map(async (file) => {
    const res = await fetch(new URL(file, new URL(MESH_BASE_URL, window.location.href)).toString());
    if (!res.ok) {
      missing.push(`${file} (${res.status})`);
      return;
    }
    mujoco.FS.writeFile(`/working/meshes/${file}`, new Uint8Array(await res.arrayBuffer()));
  }));
  if (missing.length > 0) {
    throw new Error(`Missing mesh assets: ${missing.join(', ')}`);
  }
}

function normalizeMuJoCoXml(text: string): string {
  return text.replace(/\bmeshdir=(["']).*?\1/g, 'meshdir="meshes"');
}

function mkdirp(mujoco: MujocoModule, path: string) {
  const parts = path.split('/').filter(Boolean);
  let current = '';
  for (const part of parts) {
    current += `/${part}`;
    try {
      mujoco.FS.mkdir(current);
    } catch (err: any) {
      try {
        (mujoco.FS as any).stat(current);
      } catch {
        throw err;
      }
    }
  }
}

function dirname(path: string): string {
  const idx = path.lastIndexOf('/');
  return idx <= 0 ? '/' : path.slice(0, idx);
}

function basename(path: string): string {
  return path.split('/').filter(Boolean).pop() ?? path;
}

function resolveJointSlots(mujoco: MujocoModule, model: any): JointSlots {
  const floatingId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'floating_base_joint');
  const ballId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, 'ball_freejoint');
  if (floatingId < 0 || ballId < 0) {
    throw new Error('Expected floating_base_joint and ball_freejoint in MuJoCo model');
  }
  const floatingQposAdr = model.jnt_qposadr[floatingId];
  const floatingDofAdr = model.jnt_dofadr[floatingId];
  return {
    floatingQposAdr,
    floatingDofAdr,
    robotQposAdr: floatingQposAdr + 7,
    robotDofAdr: floatingDofAdr + 6,
    ballQposAdr: model.jnt_qposadr[ballId],
    ballDofAdr: model.jnt_dofadr[ballId],
  };
}

function applyTrajectoryFrame(
  mujoco: MujocoModule,
  scene: MuJoCoScene,
  slots: JointSlots,
  traj: PingPongTrajectory,
  i: number,
) {
  scene.data.qpos[slots.floatingQposAdr + 0] = traj.base_pos[i][0];
  scene.data.qpos[slots.floatingQposAdr + 1] = traj.base_pos[i][1];
  scene.data.qpos[slots.floatingQposAdr + 2] = traj.base_pos[i][2];
  scene.data.qpos[slots.floatingQposAdr + 3] = traj.base_quat[i][0];
  scene.data.qpos[slots.floatingQposAdr + 4] = traj.base_quat[i][1];
  scene.data.qpos[slots.floatingQposAdr + 5] = traj.base_quat[i][2];
  scene.data.qpos[slots.floatingQposAdr + 6] = traj.base_quat[i][3];

  for (let j = 0; j < traj.qj[i].length; j++) {
    scene.data.qpos[slots.robotQposAdr + j] = traj.qj[i][j];
    scene.data.qvel[slots.robotDofAdr + j] = traj.dqj[i]?.[j] ?? 0;
  }

  scene.data.qpos[slots.ballQposAdr + 0] = traj.ball_pos[i][0];
  scene.data.qpos[slots.ballQposAdr + 1] = traj.ball_pos[i][1];
  scene.data.qpos[slots.ballQposAdr + 2] = traj.ball_pos[i][2];
  scene.data.qpos[slots.ballQposAdr + 3] = 1;
  scene.data.qpos[slots.ballQposAdr + 4] = 0;
  scene.data.qpos[slots.ballQposAdr + 5] = 0;
  scene.data.qpos[slots.ballQposAdr + 6] = 0;
  for (let k = 0; k < 6; k++) scene.data.qvel[slots.ballDofAdr + k] = 0;

  mujoco.mj_forward(scene.model, scene.data);
}

function buildUI(traj: PingPongTrajectory): UI {
  const hud = document.createElement('div');
  hud.className = 'hud';
  hud.innerHTML = `
    <div class="topbar">
      <div class="title">PingPongFSM Sim2Sim</div>
      <div class="subtitle">G1 · MuJoCo WASM · recorded ${traj.meta.recorded_duration_s ?? duration(traj)}s rollout</div>
    </div>
    <div></div>
    <div class="bottom">
      <div class="panel controls">
        <div class="row">
          <button class="button active" id="replay-btn" type="button">Replay</button>
          <button class="button" id="physics-btn" type="button">Physics</button>
          <span class="muted" id="physics-status">loading policy...</span>
        </div>
        <div class="row">
          <button class="button active" id="play-btn" type="button">Pause</button>
          <button class="button" id="reset-btn" type="button">Reset</button>
          <button class="button" id="serve-btn" type="button">Serve</button>
          <span class="muted">sim2sim web</span>
        </div>
        <div class="row">
          <label for="time-input">Time</label>
          <span class="readout" id="time-text">0.00s</span>
          <input class="range" id="time-input" type="range" min="0" max="${duration(traj)}" step="0.02" value="0" />
          <span class="muted" id="duration-text">${duration(traj).toFixed(2)}s</span>
        </div>
        <div class="row">
          <label for="speed-input">Speed</label>
          <input class="range" id="speed-input" type="range" min="0.1" max="2.5" step="0.1" value="1" />
          <span class="readout" id="speed-text">1.0x</span>
        </div>
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:#f8c537"></span>ball trail</span>
          <span class="legend-item"><span class="swatch" style="background:#ee4b5a"></span>hit point</span>
          <span class="legend-item"><span class="swatch" style="background:#28c99e"></span>racket</span>
          <span class="legend-item"><span class="swatch" style="background:#4e9cff"></span>base target</span>
        </div>
      </div>
      <div class="panel">
        <div class="status-grid">
          <div class="metric"><span>FSM</span><strong id="fsm-text">--</strong></div>
          <div class="metric"><span>Planner</span><strong id="planner-text">--</strong></div>
          <div class="metric"><span>Hit point</span><strong id="hit-text">--</strong></div>
          <div class="metric"><span>Ball</span><strong id="ball-text">--</strong></div>
          <div class="metric"><span>Base target</span><strong id="base-text">--</strong></div>
          <div class="metric"><span>Contact</span><strong id="contact-text">--</strong></div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('app')!.appendChild(hud);

  return {
    playBtn: hud.querySelector('#play-btn')!,
    resetBtn: hud.querySelector('#reset-btn')!,
    serveBtn: hud.querySelector('#serve-btn')!,
    replayBtn: hud.querySelector('#replay-btn')!,
    physicsBtn: hud.querySelector('#physics-btn')!,
    speedInput: hud.querySelector('#speed-input')!,
    speedText: hud.querySelector('#speed-text')!,
    timeInput: hud.querySelector('#time-input')!,
    timeText: hud.querySelector('#time-text')!,
    durationText: hud.querySelector('#duration-text')!,
    fsmText: hud.querySelector('#fsm-text')!,
    plannerText: hud.querySelector('#planner-text')!,
    hitText: hud.querySelector('#hit-text')!,
    ballText: hud.querySelector('#ball-text')!,
    baseText: hud.querySelector('#base-text')!,
    contactText: hud.querySelector('#contact-text')!,
    physicsStatus: hud.querySelector('#physics-status')!,
  };
}

function formatVec(v?: number[] | null): string {
  if (!finiteVec(v)) return '--';
  return `${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)}`;
}

function formatVec2(v?: number[] | null): string {
  if (!Array.isArray(v) || v.length < 2) return '--';
  return `${v[0].toFixed(2)}, ${v[1].toFixed(2)}`;
}

function createOverlay(scene: THREE.Scene) {
  const hitMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0xee4b5a, emissive: 0x4a080c }),
  );
  scene.add(hitMarker);

  const racketMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 20, 12),
    new THREE.MeshStandardMaterial({ color: 0x28c99e, emissive: 0x053c31 }),
  );
  scene.add(racketMarker);

  const baseTarget = new THREE.Mesh(
    new THREE.RingGeometry(0.08, 0.105, 40),
    new THREE.MeshBasicMaterial({ color: 0x4e9cff, side: THREE.DoubleSide }),
  );
  baseTarget.rotation.x = -Math.PI / 2;
  scene.add(baseTarget);

  const trailGeometry = new THREE.BufferGeometry();
  const trailMaterial = new THREE.LineBasicMaterial({ color: 0xf8c537, transparent: true, opacity: 0.9 });
  const ballTrail = new THREE.Line(trailGeometry, trailMaterial);
  scene.add(ballTrail);

  const velocityArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 0),
    0.35,
    0xf8c537,
    0.09,
    0.045,
  );
  scene.add(velocityArrow);

  return { hitMarker, racketMarker, baseTarget, ballTrail, velocityArrow };
}

function updateOverlay(
  overlay: ReturnType<typeof createOverlay>,
  traj: PingPongTrajectory,
  i: number,
) {
  const hit = traj.hit_pos_w[i];
  overlay.hitMarker.visible = finiteVec(hit);
  if (overlay.hitMarker.visible) toThree(hit, overlay.hitMarker.position);

  const racket = traj.racket_pos_w[i];
  overlay.racketMarker.visible = finiteVec(racket);
  if (overlay.racketMarker.visible) toThree(racket, overlay.racketMarker.position);

  const target = traj.base_pos_target[i];
  overlay.baseTarget.visible = Array.isArray(target) && target.length >= 2;
  if (overlay.baseTarget.visible) overlay.baseTarget.position.set(target[0], 0.024, -target[1]);

  const start = Math.max(0, i - 120);
  const points: THREE.Vector3[] = [];
  for (let k = start; k <= i; k++) {
    if (finiteVec(traj.ball_pos[k])) points.push(toThree(traj.ball_pos[k], new THREE.Vector3()));
  }
  overlay.ballTrail.geometry.dispose();
  overlay.ballTrail.geometry = new THREE.BufferGeometry().setFromPoints(points);

  const ball = traj.ball_pos[i];
  const vel = traj.ball_vel[i];
  overlay.velocityArrow.visible = finiteVec(ball) && finiteVec(vel) && Math.hypot(vel[0], vel[1], vel[2]) > 0.01;
  if (overlay.velocityArrow.visible) {
    const origin = toThree(ball, new THREE.Vector3());
    const dir = toThree(vel, new THREE.Vector3()).normalize();
    overlay.velocityArrow.position.copy(origin);
    overlay.velocityArrow.setDirection(dir);
    overlay.velocityArrow.setLength(0.35, 0.09, 0.045);
  }
}

function updateOverlayLive(
  overlay: ReturnType<typeof createOverlay>,
  sample: PhysicsSample,
) {
  const hit = sample.planner.hitPos;
  overlay.hitMarker.visible = sample.planner.valid && finiteVec(hit);
  if (overlay.hitMarker.visible) toThree(hit, overlay.hitMarker.position);

  overlay.racketMarker.visible = false;

  const target = sample.planner.baseTarget;
  overlay.baseTarget.visible = sample.planner.valid && Array.isArray(target) && target.length >= 2;
  if (overlay.baseTarget.visible) overlay.baseTarget.position.set(target[0], 0.024, -target[1]);

  const points = [toThree(sample.ballPos, new THREE.Vector3())];
  overlay.ballTrail.geometry.dispose();
  overlay.ballTrail.geometry = new THREE.BufferGeometry().setFromPoints(points);

  overlay.velocityArrow.visible = Math.hypot(sample.ballVel[0], sample.ballVel[1], sample.ballVel[2]) > 0.01;
  if (overlay.velocityArrow.visible) {
    const origin = toThree(sample.ballPos, new THREE.Vector3());
    const dir = toThree(sample.ballVel, new THREE.Vector3()).normalize();
    overlay.velocityArrow.position.copy(origin);
    overlay.velocityArrow.setDirection(dir);
    overlay.velocityArrow.setLength(0.35, 0.09, 0.045);
  }
}

function updateStatus(ui: UI, traj: PingPongTrajectory, i: number) {
  ui.timeText.textContent = `${traj.t[i].toFixed(2)}s`;
  ui.timeInput.value = String(traj.t[i]);
  ui.fsmText.textContent = fsmName(traj.fsm_state[i]);
  ui.plannerText.textContent = traj.planner_valid[i]
    ? `OK · ${traj.time_to_hit_s[i].toFixed(2)}s`
    : `reason ${traj.planner_reason_code[i]}`;
  ui.hitText.textContent = formatVec(traj.hit_pos_w[i]);
  ui.ballText.textContent = formatVec(traj.ball_pos[i]);
  ui.baseText.textContent = formatVec2(traj.base_pos_target[i]);
  ui.contactText.textContent = traj.racket_ball_contact[i] ? 'racket contact' : 'none';
}

function updateLiveStatus(ui: UI, sample: PhysicsSample) {
  ui.timeText.textContent = `${sample.simTime.toFixed(2)}s`;
  ui.fsmText.textContent = 'HITPLANNER';
  ui.plannerText.textContent = sample.planner.valid
    ? `OK · ${sample.planner.timeToHit.toFixed(2)}s`
    : sample.planner.reason;
  ui.hitText.textContent = formatVec(sample.planner.hitPos);
  ui.ballText.textContent = formatVec(sample.ballPos);
  ui.baseText.textContent = formatVec2(sample.planner.baseTarget);
  ui.contactText.textContent = sample.contact ? `racket · rally ${sample.rally}` : `rally ${sample.rally}`;
}

async function init() {
  const app = document.getElementById('app')!;
  app.innerHTML = '<div class="loading">Loading PingPongFSM sim2sim demo...</div>';

  const [mujoco, traj] = await Promise.all([
    loadMujoco({ locateFile: (path: string) => (path === 'mujoco.wasm' ? wasmUrl : path) }),
    loadTrajectory(TRAJECTORY_URL),
  ]);
  await setupVFS(mujoco);

  app.innerHTML = '<div class="viewport" id="viewport"></div>';
  const viewport = document.getElementById('viewport')!;

  const mjScene = await loadMuJoCoScene(mujoco, VFS_SCENE_PATH);
  const slots = resolveJointSlots(mujoco, mjScene.model);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x151a1f);
  scene.fog = new THREE.Fog(0x151a1f, 14, 42);
  scene.add(mjScene.root);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  viewport.appendChild(renderer.domElement);

  const camera = new THREE.PerspectiveCamera(44, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.set(4.1, 2.25, 3.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(1.55, 0.95, 0.05);
  controls.enableDamping = true;
  controls.dampingFactor = 0.055;

  const ambient = new THREE.AmbientLight(0xffffff, 0.48);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xfff1d6, 2.7);
  key.position.set(-3.5, 6.5, 3.5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 28;
  key.shadow.camera.left = -8;
  key.shadow.camera.right = 8;
  key.shadow.camera.top = 8;
  key.shadow.camera.bottom = -8;
  scene.add(key);

  const fill = new THREE.HemisphereLight(0xb9d9ff, 0x756650, 0.58);
  scene.add(fill);

  mjScene.root.traverse((obj) => {
    if ((obj as THREE.Mesh).isMesh) {
      (obj as THREE.Mesh).frustumCulled = false;
    }
  });

  const overlay = createOverlay(scene);
  const ui = buildUI(traj);

  let playing = true;
  let speed = 1;
  let currentTime = 0;
  let lastFrame = performance.now();
  let mode: 'replay' | 'physics' = 'replay';
  let physics: PhysicsController | null = null;
  let physicsReady = false;

  PhysicsController.create(mujoco, mjScene, slots).then((controller) => {
    physics = controller;
    physics.reset();
    physicsReady = true;
    ui.physicsStatus.textContent = 'policy ready';
  }).catch((err) => {
    console.error('Physics policy failed:', err);
    ui.physicsStatus.textContent = 'policy failed';
  });

  function seek(time: number) {
    currentTime = Math.max(0, Math.min(duration(traj), time));
    const i = sampleIndex(traj, currentTime);
    applyTrajectoryFrame(mujoco, mjScene, slots, traj, i);
    updateSceneTransforms(mjScene.model, mjScene.data, mjScene.bodies);
    updateOverlay(overlay, traj, i);
    updateStatus(ui, traj, i);
  }

  ui.playBtn.onclick = () => {
    playing = !playing;
    ui.playBtn.textContent = playing ? 'Pause' : 'Play';
    ui.playBtn.classList.toggle('active', playing);
  };
  ui.resetBtn.onclick = () => {
    if (mode === 'physics') {
      physics?.reset();
    } else {
      seek(0);
    }
  };
  ui.serveBtn.onclick = () => {
    mode = 'physics';
    updateModeButtons();
    physics?.serve();
  };
  ui.replayBtn.onclick = () => {
    mode = 'replay';
    updateModeButtons();
    seek(currentTime);
  };
  ui.physicsBtn.onclick = () => {
    mode = 'physics';
    updateModeButtons();
  };
  ui.speedInput.oninput = () => {
    speed = Number(ui.speedInput.value);
    ui.speedText.textContent = `${speed.toFixed(1)}x`;
  };
  ui.timeInput.oninput = () => {
    playing = false;
    ui.playBtn.textContent = 'Play';
    ui.playBtn.classList.remove('active');
    mode = 'replay';
    updateModeButtons();
    seek(Number(ui.timeInput.value));
  };

  function updateModeButtons() {
    ui.replayBtn.classList.toggle('active', mode === 'replay');
    ui.physicsBtn.classList.toggle('active', mode === 'physics');
    ui.timeInput.disabled = mode === 'physics';
  }

  seek(0);

  async function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;

    if (mode === 'replay' && playing) {
      currentTime += dt * speed;
      if (currentTime > duration(traj)) currentTime = 0;
      seek(currentTime);
    } else if (mode === 'physics' && playing && physicsReady && physics) {
      const sample = await physics.step(dt * speed);
      (window as any).__pingpongLastPhysics = sample;
      updateSceneTransforms(mjScene.model, mjScene.data, mjScene.bodies);
      updateOverlayLive(overlay, sample);
      updateLiveStatus(ui, sample);
    }

    controls.update();
    renderer.render(scene, camera);
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  animate();
}

init().catch((err) => {
  console.error(err);
  document.body.innerHTML = `
    <div style="padding:24px;color:#ffb4b4;background:#111417;font-family:system-ui,sans-serif">
      <h1 style="font-size:20px">PingPongFSM demo failed to load</h1>
      <pre>${err?.message ?? err}</pre>
    </div>
  `;
});
