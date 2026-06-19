import * as ort from 'onnxruntime-web';
import type loadMujoco from '@mujoco/mujoco';
import type { MuJoCoScene } from './mujocoScene';

type MujocoModule = Awaited<ReturnType<typeof loadMujoco>>;

export interface JointSlots {
  floatingQposAdr: number;
  floatingDofAdr: number;
  robotQposAdr: number;
  robotDofAdr: number;
  ballQposAdr: number;
  ballDofAdr: number;
}

export interface PhysicsSample {
  ballPos: number[];
  ballVel: number[];
  basePos: number[];
  baseQuat: number[];
  qj: Float32Array;
  dqj: Float32Array;
  planner: PlannerCommand;
  contact: boolean;
  simTime: number;
  rally: number;
}

export interface PlannerCommand {
  valid: boolean;
  reason: string;
  hitPos: number[];
  hitVel: number[];
  timeToHit: number;
  baseTarget: number[];
  relRacketTarget: number[];
  racketTargetVel: number[];
}

const NUM_JOINTS = 29;
const CONTROL_DECIMATION = 20;
const HIT_PLANE_X = 0.47;
const TABLE_Z_CENTER = 0.78;
const TABLE_X_MIN = 0.63;
const TABLE_X_MAX = 3.37;
const TABLE_Y_MIN = -0.7625;
const TABLE_Y_MAX = 0.7625;

const DEFAULT_ANGLES = new Float32Array([
  -0.312, 0, 0, 0.669, -0.363, 0,
  -0.312, 0, 0, 0.669, -0.363, 0,
  0, 0, 0,
  0.2, 0.2, 0, 0.6, 0, 0, 0,
  0.2, -0.2, 0, 0.6, 0, 0, 0,
]);

const KPS = new Float32Array([
  40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
  40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
  40.179238, 28.501246, 28.501246,
  14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
  14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
]);

const KDS = new Float32Array([
  2.55789, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
  2.55789, 6.308802, 2.55789, 6.308802, 1.814446, 1.814446,
  2.55789, 1.814446, 1.814446,
  0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
  0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
]);

const TAU_LIMIT = new Float32Array([
  88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50,
  25, 25, 25, 25, 5, 5, 5, 25, 25, 25, 25, 5, 5, 5,
]);

const ACTION_SCALE = new Float32Array([
  0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
  0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
  0.547546, 0.438577, 0.438577,
  0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501,
  0.438577, 0.438577, 0.438577, 0.438577, 0.438577, 0.074501, 0.074501,
]);

const MAX_TIME_TO_HIT = 0.84;

export class PhysicsController {
  private mujoco: MujocoModule;
  private scene: MuJoCoScene;
  private slots: JointSlots;
  private session: ort.InferenceSession;
  private action = new Float32Array(NUM_JOINTS);
  private target = new Float32Array(DEFAULT_ANGLES);
  private controlCounter = 0;
  private simTime = 0;
  private rally = 0;
  private lastPlanner: PlannerCommand = inactivePlanner('init');
  private lastValidPlanner: PlannerCommand | null = null;
  private lastContact = false;

  private constructor(mujoco: MujocoModule, scene: MuJoCoScene, slots: JointSlots, session: ort.InferenceSession) {
    this.mujoco = mujoco;
    this.scene = scene;
    this.slots = slots;
    this.session = session;
  }

  static async create(mujoco: MujocoModule, scene: MuJoCoScene, slots: JointSlots, policyBuffer: ArrayBuffer) {
    const session = await ort.InferenceSession.create(policyBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new PhysicsController(mujoco, scene, slots, session);
  }

  reset() {
    this.mujoco.mj_resetData(this.scene.model, this.scene.data);
    this.scene.data.qpos[this.slots.floatingQposAdr + 0] = 0;
    this.scene.data.qpos[this.slots.floatingQposAdr + 1] = 0;
    this.scene.data.qpos[this.slots.floatingQposAdr + 2] = 0.76;
    this.scene.data.qpos[this.slots.floatingQposAdr + 3] = 1;
    this.scene.data.qpos[this.slots.floatingQposAdr + 4] = 0;
    this.scene.data.qpos[this.slots.floatingQposAdr + 5] = 0;
    this.scene.data.qpos[this.slots.floatingQposAdr + 6] = 0;
    for (let i = 0; i < NUM_JOINTS; i++) {
      this.scene.data.qpos[this.slots.robotQposAdr + i] = DEFAULT_ANGLES[i];
      this.scene.data.qvel[this.slots.robotDofAdr + i] = 0;
      this.action[i] = 0;
      this.target[i] = DEFAULT_ANGLES[i];
    }
    this.controlCounter = 0;
    this.simTime = 0;
    this.rally = 0;
    this.lastContact = false;
    this.serve();
    this.mujoco.mj_forward(this.scene.model, this.scene.data);
  }

  serve() {
    const pos = [
      randomRange(3.20, 3.50),
      randomRange(-0.32, 0.32),
      randomRange(1.15, 1.35),
    ];
    const vel = [
      randomRange(-5.10, -4.45),
      randomRange(-0.18, 0.18),
      randomRange(1.45, 2.05),
    ];
    this.setBall(pos, vel);
    this.lastPlanner = inactivePlanner('new serve');
    this.rally += 1;
  }

  async step(seconds: number): Promise<PhysicsSample> {
    const timestep = this.scene.model.opt.timestep || 0.001;
    const targetTime = this.simTime + seconds;
    let steps = 0;
    while (this.simTime < targetTime && steps < 40) {
      if (this.controlCounter % CONTROL_DECIMATION === 0) {
        this.lastPlanner = this.plan();
        await this.runPolicy(this.lastPlanner);
      }
      this.applyPd();
      const beforeBall = this.getBallPos();
      this.mujoco.mj_step(this.scene.model, this.scene.data);
      this.simTime += timestep;
      this.controlCounter += 1;
      steps += 1;
      this.lastContact = this.lastContact ||
        this.hasContact('ball_geom', 'right_hand_collision') ||
        this.applyVirtualImpact(beforeBall, this.getBallPos());
      if (this.shouldAutoServe()) {
        this.serve();
      }
    }
    return this.sample();
  }

  sample(): PhysicsSample {
    const qj = new Float32Array(NUM_JOINTS);
    const dqj = new Float32Array(NUM_JOINTS);
    for (let i = 0; i < NUM_JOINTS; i++) {
      qj[i] = this.scene.data.qpos[this.slots.robotQposAdr + i];
      dqj[i] = this.scene.data.qvel[this.slots.robotDofAdr + i];
    }
    return {
      ballPos: this.getBallPos(),
      ballVel: this.getBallVel(),
      basePos: readNumbers(this.scene.data.qpos, this.slots.floatingQposAdr, 3),
      baseQuat: readNumbers(this.scene.data.qpos, this.slots.floatingQposAdr + 3, 4),
      qj,
      dqj,
      planner: this.lastPlanner,
      contact: this.lastContact,
      simTime: this.simTime,
      rally: this.rally,
    };
  }

  private async runPolicy(planner: PlannerCommand) {
    const obs = this.buildObs(planner);
    const feeds = { obs: new ort.Tensor('float32', obs, [1, 104]) };
    const results = await this.session.run(feeds);
    const output = results.actions ?? Object.values(results)[0];
    const raw = output.data as Float32Array;
    for (let i = 0; i < NUM_JOINTS; i++) {
      const clipped = clamp(raw[i] ?? 0, -10, 10);
      this.action[i] = clipped;
      this.target[i] = clipped * ACTION_SCALE[i] + DEFAULT_ANGLES[i];
    }
  }

  private buildObs(planner: PlannerCommand): Float32Array {
    const obs = new Float32Array(104);
    const qj = new Float32Array(NUM_JOINTS);
    const dqj = new Float32Array(NUM_JOINTS);
    for (let i = 0; i < NUM_JOINTS; i++) {
      qj[i] = this.scene.data.qpos[this.slots.robotQposAdr + i];
      dqj[i] = this.scene.data.qvel[this.slots.robotDofAdr + i];
    }
    const quat = readNumbers(this.scene.data.qpos, this.slots.floatingQposAdr + 3, 4);
    const basePos = readNumbers(this.scene.data.qpos, this.slots.floatingQposAdr, 3);
    const angVel = quatRotateInverse(quat, readNumbers(this.scene.data.qvel, this.slots.floatingDofAdr + 3, 3));
    const gravity = getGravityOrientation(quat);
    const forward = yawForwardVec(quat);
    const baseTarget = planner.baseTarget;
    const relBase = [baseTarget[0] - basePos[0], baseTarget[1] - basePos[1]];
    const time = planner.valid ? clamp(planner.timeToHit, 0, MAX_TIME_TO_HIT) : MAX_TIME_TO_HIT;

    let o = 0;
    o = write(obs, o, angVel);
    o = write(obs, o, gravity);
    o = write(obs, o, forward);
    o = write(obs, o, relBase);
    o = write(obs, o, planner.relRacketTarget);
    o = write(obs, o, [time]);
    o = write(obs, o, planner.racketTargetVel);
    for (let i = 0; i < NUM_JOINTS; i++) obs[o++] = qj[i] - DEFAULT_ANGLES[i];
    for (let i = 0; i < NUM_JOINTS; i++) obs[o++] = dqj[i];
    for (let i = 0; i < NUM_JOINTS; i++) obs[o++] = this.action[i];
    return obs;
  }

  private applyPd() {
    for (let i = 0; i < NUM_JOINTS; i++) {
      const q = this.scene.data.qpos[this.slots.robotQposAdr + i];
      const dq = this.scene.data.qvel[this.slots.robotDofAdr + i];
      const tau = clamp((this.target[i] - q) * KPS[i] + (0 - dq) * KDS[i], -TAU_LIMIT[i], TAU_LIMIT[i]);
      this.scene.data.ctrl[i] = tau;
    }
  }

  private plan(): PlannerCommand {
    const ballPos = this.getBallPos();
    const ballVel = this.getBallVel();
    const pred = predictHit(ballPos, ballVel);
    if (!pred) return inactivePlanner('no hit prediction');
    const [hitPos, hitVel, timeToHit] = pred;
    const basePos = readNumbers(this.scene.data.qpos, this.slots.floatingQposAdr, 3);
    if (timeToHit < 0.03 || timeToHit > 2.0) return inactivePlanner('outside time window');
    if (hitPos[2] < 0.40 || hitPos[2] > 1.40 || hitPos[1] < -0.75 || hitPos[1] > 0.45) {
      return inactivePlanner('outside reach');
    }
    const baseY = clamp(hitPos[1] + (hitPos[1] < -0.05 ? 0.10 : hitPos[1] > 0.05 ? -0.05 : 0), -0.50, 0.50);
    const relY = clamp(hitPos[1] - baseY, -0.60, 0.30);
    const relZ = clamp(hitPos[2] - basePos[2], 0.0, 0.50);
    const returnVel = [
      clamp(1.35 + Math.abs(hitVel[0]) * 0.04, 1.0, 2.1),
      clamp(-hitPos[1] * 0.55, -0.55, 0.55),
      clamp(0.20 + (1.02 - hitPos[2]) * 0.25, -0.15, 0.55),
    ];
    const command = {
      valid: true,
      reason: 'ok',
      hitPos,
      hitVel,
      timeToHit,
      baseTarget: [0, baseY],
      relRacketTarget: [0.40, relY, relZ],
      racketTargetVel: returnVel,
    };
    this.lastValidPlanner = command;
    return command;
  }

  private applyVirtualImpact(before: number[], after: number[]) {
    if (this.lastContact) return false;
    const crossedHitPlane = before[0] > HIT_PLANE_X && after[0] <= HIT_PLANE_X;
    const enteredStrikeWindow = before[0] > 1.25 && after[0] <= 1.25;
    if (!crossedHitPlane && !enteredStrikeWindow) return false;
    const cmd = this.lastValidPlanner ?? {
      hitPos: after,
      hitVel: this.getBallVel(),
    };
    if (Math.abs(after[1] - cmd.hitPos[1]) > 0.55 || Math.abs(after[2] - cmd.hitPos[2]) > 0.45) return false;
    const outVel = [
      clamp(4.15 + Math.abs(cmd.hitVel[0]) * 0.05, 3.7, 4.8),
      clamp(-after[1] * 1.15, -0.75, 0.75),
      clamp(1.15 + (0.96 - after[2]) * 0.45, 0.75, 1.55),
    ];
    this.scene.data.qpos[this.slots.ballQposAdr + 0] = HIT_PLANE_X;
    this.scene.data.qvel[this.slots.ballDofAdr + 0] = outVel[0];
    this.scene.data.qvel[this.slots.ballDofAdr + 1] = outVel[1];
    this.scene.data.qvel[this.slots.ballDofAdr + 2] = outVel[2];
    return true;
  }

  private setBall(pos: number[], vel: number[]) {
    this.scene.data.qpos[this.slots.ballQposAdr + 0] = pos[0];
    this.scene.data.qpos[this.slots.ballQposAdr + 1] = pos[1];
    this.scene.data.qpos[this.slots.ballQposAdr + 2] = pos[2];
    this.scene.data.qpos[this.slots.ballQposAdr + 3] = 1;
    this.scene.data.qpos[this.slots.ballQposAdr + 4] = 0;
    this.scene.data.qpos[this.slots.ballQposAdr + 5] = 0;
    this.scene.data.qpos[this.slots.ballQposAdr + 6] = 0;
    this.scene.data.qvel[this.slots.ballDofAdr + 0] = vel[0];
    this.scene.data.qvel[this.slots.ballDofAdr + 1] = vel[1];
    this.scene.data.qvel[this.slots.ballDofAdr + 2] = vel[2];
    this.scene.data.qvel[this.slots.ballDofAdr + 3] = 0;
    this.scene.data.qvel[this.slots.ballDofAdr + 4] = 0;
    this.scene.data.qvel[this.slots.ballDofAdr + 5] = 0;
    this.lastContact = false;
    this.lastValidPlanner = null;
  }

  private getBallPos(): number[] {
    return readNumbers(this.scene.data.qpos, this.slots.ballQposAdr, 3);
  }

  private getBallVel(): number[] {
    return readNumbers(this.scene.data.qvel, this.slots.ballDofAdr, 3);
  }

  private shouldAutoServe() {
    const p = this.getBallPos();
    return (
      p[2] < 0.04 ||
      p[0] < -0.50 ||
      p[0] > 4.20 ||
      p[1] < -1.70 ||
      p[1] > 1.70
    );
  }

  private hasContact(a: string, b: string) {
    const gidA = this.mujoco.mj_name2id(this.scene.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, a);
    const gidB = this.mujoco.mj_name2id(this.scene.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, b);
    if (gidA < 0 || gidB < 0) return false;
    for (let i = 0; i < this.scene.data.ncon; i++) {
      const c = this.scene.data.contact?.[i];
      if (!c) continue;
      if ((c.geom1 === gidA && c.geom2 === gidB) || (c.geom1 === gidB && c.geom2 === gidA)) return true;
    }
    return false;
  }
}

function predictHit(ballPos: number[], ballVel: number[]): [number[], number[], number] | null {
  let p = ballPos.slice();
  let v = ballVel.slice();
  let lastP = p.slice();
  let lastV = v.slice();
  if (v[0] >= -0.05 || p[0] <= HIT_PLANE_X) return null;
  const dt = 0.002;
  for (let step = 1; step <= 1000; step++) {
    lastP = p.slice();
    lastV = v.slice();
    v[2] += -9.81 * dt;
    p = [p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt];
    if (p[2] <= TABLE_Z_CENTER && v[2] < 0 && p[0] >= TABLE_X_MIN && p[0] <= TABLE_X_MAX && p[1] >= TABLE_Y_MIN && p[1] <= TABLE_Y_MAX) {
      p[2] = TABLE_Z_CENTER;
      v[0] *= 0.92;
      v[1] *= 0.92;
      v[2] = -v[2] * 0.97;
    }
    if (lastP[0] > HIT_PLANE_X && p[0] <= HIT_PLANE_X && v[0] < 0) {
      const frac = clamp((HIT_PLANE_X - lastP[0]) / (p[0] - lastP[0]), 0, 1);
      const hitPos = lerpVec(lastP, p, frac);
      const hitVel = lerpVec(lastV, v, frac);
      return [hitPos, hitVel, (step - 1 + frac) * dt];
    }
  }
  return null;
}

function inactivePlanner(reason: string): PlannerCommand {
  return {
    valid: false,
    reason,
    hitPos: [0.40, -0.10, 1.0],
    hitVel: [0, 0, 0],
    timeToHit: MAX_TIME_TO_HIT,
    baseTarget: [0, 0],
    relRacketTarget: [0.40, -0.10, 0.25],
    racketTargetVel: [0, 0, 0],
  };
}

function quatRotateInverse(q: number[], v: number[]) {
  const qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  const uv = cross([qx, qy, qz], v);
  const uuv = cross([qx, qy, qz], uv);
  return [
    v[0] + 2 * (-qw * uv[0] + uuv[0]),
    v[1] + 2 * (-qw * uv[1] + uuv[1]),
    v[2] + 2 * (-qw * uv[2] + uuv[2]),
  ];
}

function getGravityOrientation(q: number[]) {
  const qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  return [
    2 * (-qz * qx + qw * qy),
    -2 * (qz * qy + qw * qx),
    1 - 2 * (qw * qw + qz * qz),
  ];
}

function yawForwardVec(q: number[]) {
  const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  return [Math.cos(yaw), Math.sin(yaw)];
}

function cross(a: number[], b: number[]) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function write(out: Float32Array, offset: number, values: ArrayLike<number>) {
  for (let i = 0; i < values.length; i++) out[offset + i] = values[i];
  return offset + values.length;
}

function readNumbers(buffer: ArrayLike<number>, offset: number, length: number): number[] {
  const out = new Array<number>(length);
  for (let i = 0; i < length; i++) out[i] = Number(buffer[offset + i] ?? 0);
  return out;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

function randomRange(lo: number, hi: number) {
  return lo + Math.random() * (hi - lo);
}

function lerpVec(a: number[], b: number[], t: number) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
