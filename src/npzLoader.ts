import JSZip from 'jszip';
import type { PingPongTrajectory } from './trajectory';

interface NpyArray {
  shape: number[];
  data: Array<number | boolean>;
}

type NpzMap = Record<string, NpyArray>;

export async function loadTrajectoryFromFile(file: File): Promise<PingPongTrajectory> {
  if (file.name.toLowerCase().endsWith('.json')) {
    return JSON.parse(await file.text()) as PingPongTrajectory;
  }
  if (!file.name.toLowerCase().endsWith('.npz')) {
    throw new Error('Replay upload expects .npz or converted .json');
  }
  const zip = await JSZip.loadAsync(await file.arrayBuffer());
  const arrays: NpzMap = {};
  await Promise.all(Object.values(zip.files).map(async (entry) => {
    if (entry.dir || !entry.name.endsWith('.npy')) return;
    const key = entry.name.replace(/\.npy$/, '').split('/').pop();
    if (!key) return;
    try {
      arrays[key] = parseNpy(await entry.async('arraybuffer'));
    } catch {
      // Object arrays such as pickle-backed meta_json are intentionally skipped.
    }
  }));
  return trajectoryFromArrays(arrays, file.name);
}

function parseNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer);
  if (
    bytes[0] !== 0x93 ||
    bytes[1] !== 0x4e ||
    bytes[2] !== 0x55 ||
    bytes[3] !== 0x4d ||
    bytes[4] !== 0x50 ||
    bytes[5] !== 0x59
  ) {
    throw new Error('Invalid NPY magic');
  }
  const major = bytes[6];
  const view = new DataView(buffer);
  let headerLen = 0;
  let headerStart = 0;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}`);
  }
  const header = new TextDecoder('latin1').decode(bytes.subarray(headerStart, headerStart + headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === 'True';
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || !shapeText) throw new Error('Invalid NPY header');
  if (fortran) throw new Error('Fortran-order NPY is not supported');
  const shape = shapeText
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => Number(x));
  const count = shape.reduce((a, b) => a * b, 1);
  const dataOffset = headerStart + headerLen;
  const data: Array<number | boolean> = [];
  for (let i = 0; i < count; i++) {
    data.push(readValue(view, dataOffset, i, descr));
  }
  return { shape, data };
}

function readValue(view: DataView, offset: number, index: number, descr: string): number | boolean {
  switch (descr) {
    case '<f4':
    case '|f4':
      return view.getFloat32(offset + index * 4, true);
    case '<f8':
      return view.getFloat64(offset + index * 8, true);
    case '|i1':
      return view.getInt8(offset + index);
    case '|u1':
      return view.getUint8(offset + index);
    case '|b1':
    case '?':
      return view.getUint8(offset + index) !== 0;
    case '<i2':
      return view.getInt16(offset + index * 2, true);
    case '<u2':
      return view.getUint16(offset + index * 2, true);
    case '<i4':
      return view.getInt32(offset + index * 4, true);
    case '<u4':
      return view.getUint32(offset + index * 4, true);
    default:
      throw new Error(`Unsupported NPY dtype ${descr}`);
  }
}

function trajectoryFromArrays(arrays: NpzMap, source: string): PingPongTrajectory {
  const t = numberVector(requireArray(arrays, 't'));
  const n = t.length;
  return {
    t,
    ball_pos: numberRows(requireArray(arrays, 'ball_pos'), n, 3),
    ball_vel: numberRows(requireArray(arrays, 'ball_vel'), n, 3),
    base_pos: numberRows(requireArray(arrays, 'base_pos'), n, 3),
    base_quat: numberRows(requireArray(arrays, 'base_quat'), n, 4),
    qj: numberRows(requireArray(arrays, 'qj'), n, 29),
    dqj: numberRows(arrays.dqj, n, 29, 0),
    fsm_state: intVector(arrays.fsm_state, n, 13),
    supervisor_downgraded: boolVector(arrays.supervisor_downgraded, n, false),
    planner_status: intVector(arrays.planner_status, n, 0),
    hit_pos_w: numberRows(arrays.hit_pos_w, n, 3, Number.NaN),
    time_to_hit_s: numberVector(arrays.time_to_hit_s, n, Number.NaN),
    base_pos_target: numberRows(arrays.base_pos_target, n, 2, Number.NaN),
    planner_valid: boolVector(arrays.planner_valid, n, false),
    planner_reason_code: intVector(arrays.planner_reason_code, n, -1),
    rel_racket_target_pos_w: numberRows(arrays.rel_racket_target_pos_w, n, 3, Number.NaN),
    racket_target_vel_w: numberRows(arrays.racket_target_vel_w, n, 3, Number.NaN),
    racket_pos_w: numberRows(arrays.racket_pos_w, n, 3, Number.NaN),
    racket_ball_contact: boolVector(arrays.racket_ball_contact, n, false),
    meta: {
      source_file: source,
      recorded_duration_s: t[t.length - 1] ?? 0,
      recorded_ticks: n,
      uploaded_npz: true,
    },
    source,
  };
}

function requireArray(arrays: NpzMap, key: string): NpyArray {
  const value = arrays[key];
  if (!value) throw new Error(`NPZ missing required array: ${key}`);
  return value;
}

function numberVector(arr?: NpyArray, length?: number, fill = 0): number[] {
  if (!arr) return new Array(length ?? 0).fill(fill);
  return (arr.data as Array<number | boolean>).map((v) => Number(v));
}

function intVector(arr: NpyArray | undefined, length: number, fill: number): number[] {
  if (!arr) return new Array(length).fill(fill);
  return (arr.data as Array<number | boolean>).slice(0, length).map((v) => Number(v) | 0);
}

function boolVector(arr: NpyArray | undefined, length: number, fill: boolean): boolean[] {
  if (!arr) return new Array(length).fill(fill);
  return (arr.data as Array<number | boolean>).slice(0, length).map((v) => Boolean(v));
}

function numberRows(arr: NpyArray | undefined, rows: number, cols: number, fill = 0): number[][] {
  if (!arr) return Array.from({ length: rows }, () => new Array(cols).fill(fill));
  const data = arr.data as Array<number | boolean>;
  const out: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push(Number(data[r * cols + c] ?? fill));
    }
    out.push(row);
  }
  return out;
}
