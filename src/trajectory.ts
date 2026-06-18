export interface PingPongTrajectory {
  t: number[];
  ball_pos: number[][];
  ball_vel: number[][];
  base_pos: number[][];
  base_quat: number[][];
  qj: number[][];
  dqj: number[][];
  fsm_state: number[];
  supervisor_downgraded: boolean[];
  planner_status: number[];
  hit_pos_w: number[][];
  time_to_hit_s: number[];
  base_pos_target: number[][];
  planner_valid: boolean[];
  planner_reason_code: number[];
  rel_racket_target_pos_w: number[][];
  racket_target_vel_w: number[][];
  racket_pos_w: number[][];
  racket_ball_contact: boolean[];
  meta: Record<string, unknown>;
  source: string;
}

export async function loadTrajectory(url: string): Promise<PingPongTrajectory> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch trajectory: ${url} (${res.status} ${res.statusText})`);
  }
  return res.json();
}

export function sampleIndex(traj: PingPongTrajectory, time: number): number {
  const t = traj.t;
  if (t.length <= 1) return 0;
  if (time <= t[0]) return 0;
  const last = t.length - 1;
  if (time >= t[last]) return last;
  const dt = t[1] - t[0] || 0.02;
  return Math.max(0, Math.min(last, Math.round(time / dt)));
}

export function duration(traj: PingPongTrajectory): number {
  return traj.t[traj.t.length - 1] ?? 0;
}

export function fsmName(value: number): string {
  const names: Record<number, string> = {
    0: 'PASSIVE',
    1: 'FIXEDPOSE',
    2: 'LOCOMODE',
    10: 'SKILL_COOLDOWN',
    11: 'HRLHIT_ISAAC',
    12: 'HRLHIT_MJ',
    13: 'HITPLANNER',
  };
  return names[value] ?? `STATE_${value}`;
}
