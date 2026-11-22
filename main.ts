// Push-optimal Sokoban solver: the hallway is cut out of the floor, each
// room holding boxes is solved on its own by push-optimal A*, and the rooms'
// optima add up to the level's. Pruning is safe (live squares by reverse
// pull from the goals, frozen 2x2 blocks); the heuristic is the minimum
// assignment of Manhattan box-to-goal distances, so A* stays optimal.
// Deterministic: no clock, no randomness; budgets are counted states.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const D = [
  { x: 0, y: -1, w: 'u', p: 'U' },
  { x: 0, y: 1, w: 'd', p: 'D' },
  { x: -1, y: 0, w: 'l', p: 'L' },
  { x: 1, y: 0, w: 'r', p: 'R' },
];
const CAP_R = 300_000;
const CAP_L = 250_000;
type L = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };
function parse(t: string): L {
  const rows = t.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const c = row[x];
      const i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goals.push(i);
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}
function live(l: L): Uint8Array {
  const { w, wall, goals } = l;
  const v = new Uint8Array(wall.length);
  const q = [...goals];
  for (const g of goals) v[g] = 1;
  for (let i = 0; i < q.length; i++) {
    const x = q[i];
    for (const d of D) {
      const dd = d.y * w + d.x;
      const y = x - dd;
      if (!wall[y] && !wall[y - dd] && !v[y]) { v[y] = 1; q.push(y); }
    }
  }
  return v;
}
function reach(l: L, occ: Uint8Array, from: number, seen: Int32Array, st: number, sk: Int32Array): number {
  const { w, wall } = l;
  let low = from;
  let n = 0;
  sk[n++] = from;
  seen[from] = st;
  while (n > 0) {
    const p = sk[--n];
    if (p < low) low = p;
    for (const d of D) {
      const q = p + d.y * w + d.x;
      if (wall[q] === 0 && occ[q] === 0 && seen[q] !== st) { seen[q] = st; sk[n++] = q; }
    }
  }
  return low;
}
function frozen(l: L, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = l;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b])) continue;
    if (!(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}
function walk(l: L, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = l;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i];
    if (p === to) break;
    for (const d of D) {
      const s = p + d.y * w + d.x;
      if (wall[s] || occ[s] || prev[s] !== -1) continue;
      prev[s] = p; how[s] = d.w; q.push(s);
    }
  }
  if (prev[to] === -1) throw new Error('walk: unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const c of moves.toLowerCase()) p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  return p;
}
function pc(x: number): number { let c = 0; while (x) { x &= x - 1; c++; } return c; }
function solve(l: L, cap: number): { pushes: number; moves: string } | null {
  const { w, wall, goals } = l;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const lv = live(l);
  if (l.boxes.some((b) => !lv[b])) return null;
  const done = (b: number[]): boolean => b.every((x) => goal[x]);
  const m = goals.length;
  const bx = (i: number): number => i % w;
  const by = (i: number): number => Math.floor(i / w);
  const hOf = (boxes: number[]): number => {
    const n = boxes.length;
    if (n === 0) return 0;
    if (n > 8 || m > 8) {
      let s = 0;
      for (const b of boxes) {
        let best = Infinity;
        for (const g of goals) { const d = Math.abs(bx(b) - bx(g)) + Math.abs(by(b) - by(g)); if (d < best) best = d; }
        s += best;
      }
      return s;
    }
    const dm = new Int32Array(n * m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) dm[i * m + j] = Math.abs(bx(boxes[i]) - bx(goals[j])) + Math.abs(by(boxes[i]) - by(goals[j]));
    const st = 1 << m;
    const dp = new Float64Array(st).fill(Infinity);
    dp[0] = 0;
    for (let mask = 0; mask < st; mask++) {
      const i = pc(mask);
      if (i >= n || dp[mask] === Infinity) continue;
      for (let j = 0; j < m; j++) {
        if (mask & (1 << j)) continue;
        const nm = mask | (1 << j);
        const v = dp[mask] + dm[i * m + j];
        if (v < dp[nm]) dp[nm] = v;
      }
    }
    let best = Infinity;
    for (let mask = 0; mask < st; mask++) if (pc(mask) === n && dp[mask] < best) best = dp[mask];
    return best;
  };
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const sk = new Int32Array(size);
  let stamp = 0;
  const num = l.boxes.length + 1 <= 5;
  const key = (boxes: number[], low: number): number | string => {
    if (!num) return low + ':' + boxes.join(',');
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };
  const sB: number[][] = [l.boxes.slice()];
  const sP: number[] = [l.player];
  const par: number[] = [-1];
  const pD: number[] = [-1];
  const keys: (number | string)[] = [];
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const push = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) { const p = (i - 1) >> 1; if (!less(heap[i]!, heap[p]!)) break; const t = heap[i]!; heap[i] = heap[p]!; heap[p] = t; i = p; }
  };
  const pop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!; heap[i] = heap[c]!; heap[c] = t; i = c;
      }
    }
    return top;
  };
  const best = new Map<number | string, number>();
  for (const b of l.boxes) occ[b] = 1;
  const low0 = reach(l, occ, l.player, seen, ++stamp, sk);
  for (const b of l.boxes) occ[b] = 0;
  const k0 = key(l.boxes, low0);
  best.set(k0, 0);
  keys.push(k0);
  push([hOf(l.boxes), 0, seq++, 0]);
  let found = done(l.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, ng, , s] = pop();
    const g = -ng;
    const boxes = sB[s]!;
    if (g > (best.get(keys[s]!) ?? Infinity)) continue;
    for (const b of boxes) occ[b] = 1;
    reach(l, occ, sP[s]!, seen, ++stamp, sk);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < D.length; di++) {
        const dir = D[di]!;
        const d = dir.y * w + dir.x;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !lv[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const fr = frozen(l, occ, goal, t);
        let k: number | string = '';
        let nx: number[] | null = null;
        if (!fr) {
          nx = boxes.slice(); nx[bi] = t; nx.sort((a, b2) => a - b2);
          k = key(nx, reach(l, occ, b, mark, ++stamp, sk));
        }
        occ[t] = 0; occ[b] = 1;
        if (fr || nx === null) continue;
        const g2 = g + 1;
        const pb = best.get(k);
        if (pb !== undefined && pb <= g2) continue;
        if (sB.length >= cap) { for (const x of boxes) occ[x] = 0; return null; }
        best.set(k, g2);
        const ns = sB.length;
        sB.push(nx); sP.push(b); par.push(s); pD.push(di); keys.push(k);
        if (done(nx)) { found = ns; break; }
        push([g2 + hOf(nx), -g2, seq++, ns]);
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = par[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = l.player;
  let boxes = l.boxes.slice();
  for (const s of path) {
    const nb = sB[s]!;
    const from = boxes.find((b) => !nb.includes(b))!;
    const to = nb.find((b) => !boxes.includes(b))!;
    const dir = D[pD[s]!]!;
    const d = dir.y * w + dir.x;
    if (to - from !== d) throw new Error('reconstruct');
    for (const b of boxes) occ[b] = 1;
    moves += walk(l, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += dir.p;
    player = from;
    boxes = nb;
  }
  return { pushes: path.length, moves };
}
type R = { boxes: number[]; goals: number[]; done: boolean };
function rooms(l: L): R[] | null {
  const { w, wall, goals, boxes, player } = l;
  const lv = live(l);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !lv[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rs: R[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const sq = [s];
    part[s] = rs.length;
    for (let i = 0; i < sq.length; i++) for (const d of D) {
      const q = sq[i] + d.y * w + d.x;
      if (!wall[q] && !cut[q] && part[q] < 0) { part[q] = rs.length; sq.push(q); }
    }
    rs.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rs[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rs[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rs.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rs.filter((r) => r.boxes.length > 0);
}
function solveRooms(l: L): { pushes: number; moves: string } | null {
  const rs = rooms(l);
  if (rs === null) return solve(l, CAP_L);
  let player = l.player;
  let pushes = 0;
  let moves = '';
  for (const r of rs) {
    const wall = l.wall.slice();
    for (const o of rs) if (o !== r) for (const b of o.done ? o.goals : o.boxes) wall[b] = 1;
    const sub: L = { w: l.w, h: l.h, wall, goals: r.goals.slice().sort((a, b) => a - b), boxes: r.boxes.slice().sort((a, b) => a - b), player };
    const s = solve(sub, CAP_R);
    if (s === null) return solve(l, CAP_L);
    pushes += s.pushes;
    moves += s.moves;
    player = endOf(l.w, player, s.moves);
    r.done = true;
  }
  return { pushes, moves };
}
function solveLevel(t: string): string {
  try {
    const s = solveRooms(parse(t));
    return s === null ? '' : s.moves;
  } catch {
    return '';
  }
}
for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
