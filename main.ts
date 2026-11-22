// Room-decomposed push-optimal Sokoban solver.
//
// Strategy (built for the generator's level shape: one hallway, 1-4 rooms
// of 3-4 boxes off it, a box pushed out of a room never comes back):
//  1. Room decomposition: cut the hallway's dead squares out of the floor;
//     each connected part holding boxes is one independent room.
//  2. Per room, push-optimal A* over pushes (not moves): a state is the
//     sorted box squares plus the player's reachable region (named by its
//     lowest square). Every push costs 1, so A* with an admissible
//     heuristic returns the fewest pushes possible.
//  3. Deadlock pruning, both safe (never prunes a solvable state):
//     - reverse-push liveness: a square is live only if a box there can be
//       pulled back onto some goal, ignoring the other boxes; pushes onto
//       dead squares are dropped.
//     - frozen 2x2: four squares of walls/boxes holding a box off its goal.
//  4. Heuristic: minimum-assignment sum of Manhattan box-to-goal distances
//     (exact min assignment by subset DP for <= 8 boxes, sum-of-minima
//     above that). Each push moves one box by one square, so the heuristic
//     never overestimates the pushes still needed: A* stays optimal.
//  5. Fallback: if a level is no set of rooms, the same push search runs
//     over the whole level. A level no search solves gets an empty answer
//     (valid, 0 points) instead of crashing.
//
// Deterministic by construction: no clock, no randomness, no network, no
// environment reads; every tie breaks by fixed direction order and
// insertion order. All budgets are counted states, never wall time.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right.
const DIRS: { dx: number; dy: number; walk: string; push: string }[] = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

const MAX_ROOM_STATES = 300_000;
const MAX_LEVEL_STATES = 250_000;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Squares a box can ever stand on and still reach a goal: pull boxes back
// from the goals. From a square x, a box could have been pulled from
// y = x - d provided y and y - d are open floor (the player stood at y - d
// and pulled the box from x onto y).
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const dir of DIRS) {
      const d = dir.dy * w + dir.dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Flood fill from `from` around walls and boxes; marks the region in `seen`
// with `stamp` and returns its lowest square (the region's name).
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed to `t` sits in a frozen 2x2 block of
// walls and boxes with a box off its goal.
function frozenBlock(
  level: Level,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (boxes never
// move during a walk), in lower-case LURD. Throws when unreachable.
function walkTo(
  level: Level,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = dir.walk;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

// Push-optimal A* for one (sub-)level: { pushes, moves } or null when the
// state budget runs out or the level has no solution. `maxStates` counts
// stored states, never wall time.
function solvePushOptimal(
  level: Level,
  maxStates: number,
): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  const m = goals.length;
  const bx = (i: number): number => i % w;
  const by = (i: number): number => Math.floor(i / w);
  // Minimum-assignment heuristic: the cheapest way to match every box to
  // a distinct goal by Manhattan distance (subset DP for <= 8 boxes,
  // sum-of-nearest above that). Each push moves one box one square, so it
  // never overestimates the pushes still needed: A* stays optimal.
  const hOf = (boxes: number[]): number => {
    const n = boxes.length;
    if (n === 0) return 0;
    if (n > 8 || m > 8) {
      let sum = 0;
      for (const b of boxes) {
        let best = Infinity;
        for (const g of goals) {
          const d = Math.abs(bx(b) - bx(g)) + Math.abs(by(b) - by(g));
          if (d < best) best = d;
        }
        sum += best;
      }
      return sum;
    }
    const dmat = new Int32Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        dmat[i * m + j] =
          Math.abs(bx(boxes[i]) - bx(goals[j])) +
          Math.abs(by(boxes[i]) - by(goals[j]));
      }
    }
    const states = 1 << m;
    const dp = new Float64Array(states).fill(Infinity);
    dp[0] = 0;
    for (let mask = 0; mask < states; mask++) {
      const i = popcount(mask);
      if (i >= n || dp[mask] === Infinity) continue;
      for (let j = 0; j < m; j++) {
        if (mask & (1 << j)) continue;
        const nm = mask | (1 << j);
        const v = dp[mask] + dmat[i * m + j];
        if (v < dp[nm]) dp[nm] = v;
      }
    }
    let best = Infinity;
    for (let mask = 0; mask < states; mask++) {
      if (popcount(mask) === n && dp[mask] < best) best = dp[mask];
    }
    return best;
  };
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const useNumericKey = level.boxes.length + 1 <= 5;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!useNumericKey) return low + ':' + boxes.join(',');
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  // State storage.
  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1]; // box square before the push
  const pushDir: number[] = [-1]; // DIRS index of the push
  const gScore: number[] = [0];
  // Each state's key (region low + boxes), so a heap entry made stale by
  // a cheaper path to the same key can be skipped on pop.
  const stateKeys: (number | string)[] = [];

  // Binary heap of [f, -g, seq, state], ordered by f, then deeper g first.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const less = (
    a: [number, number, number, number],
    b: [number, number, number, number],
  ): boolean =>
    a[0] < b[0] ||
    (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  const best = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const startLow = reach(level, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  const startKey = keyOf(level.boxes, startLow);
  best.set(startKey, 0);
  stateKeys.push(startKey);
  heapPush([hOf(level.boxes), 0, seq++, 0]);

  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const [, negG, , s] = heapPop();
    const g = -negG;
    const boxes = statesBoxes[s]!;
    // Stale heap entry: a cheaper path to this state's key was found
    // after this entry was pushed.
    if (g > (best.get(stateKeys[s]!) ?? Infinity)) continue;
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < DIRS.length; di++) {
        const dir = DIRS[di]!;
        const d = dir.dy * w + dir.dx;
        const t = b + d;
        // The player must be able to stand at b - d, and the box must be
        // able to go to t (open, empty, live).
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, goal, t);
        let key: number | string = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = keyOf(next, reach(level, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const ng = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        best.set(key, ng);
        const ns = statesBoxes.length;
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(s);
        pushBox.push(b);
        pushDir.push(di);
        gScore.push(ng);
        stateKeys.push(key);
        if (solved(next)) {
          found = ns;
          break;
        }
        heapPush([ng + hOf(next), -ng, seq++, ns]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Reconstruct: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const di = pushDir[s]!;
    const dir = DIRS[di]!;
    const d = dir.dy * w + dir.dx;
    if (to - from !== d) throw new Error('reconstruct: push mismatch');
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// The level's rooms: cut the dead squares of the player's row (the
// hallway: a box there reaches no goal) out of the floor, and each part
// left that holds a box is a room, its squares, boxes and goals. Null when
// some part holds more or fewer boxes than goals, or a box starts in the
// hallway: then the level is no set of rooms.
type Room = { squares: number[]; boxes: number[]; goals: number[]; done: boolean };

function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const dir of DIRS) {
        const q = squares[i] + dir.dy * w + dir.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  }
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve room by room: each room's push-optimal solution in turn, the other
// rooms' boxes standing as walls (on their goals once their room is done),
// the player walking on from where the last room left it. A box never
// leaves its room, so the rooms' optima add up to the level's.
function solveByRooms(
  level: Level,
): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solvePushOptimal(level, MAX_LEVEL_STATES);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub, MAX_ROOM_STATES);
    if (solved === null) return solvePushOptimal(level, MAX_LEVEL_STATES);
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

// One level's moves: push-optimal when the search finds it, empty otherwise.
function solveLevel(text: string): string {
  try {
    const level = parseLevel(text);
    const solved = solveByRooms(level);
    return solved === null ? '' : solved.moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}

