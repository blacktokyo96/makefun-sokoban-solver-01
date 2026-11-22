# Room-decomposed push-optimal Sokoban solver

Push-optimal Sokoban solver for make.fun's
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty ($100 in USDC).

The level's hallway is cut out of the floor, each room holding boxes is
solved on its own, and the rooms' optima add up to the level's: per room,
push-optimal A* over pushes with reverse-pull liveness pruning, frozen-2x2
deadlock pruning, and a minimum-assignment Manhattan heuristic. Deterministic
(no clock, no randomness); the whole 5-instance set runs in ~1 s.

Run: `MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> node main.ts`

Practice set: **1,000,000** — every instance solved in exactly P* pushes
(13, 16, 29, 47, 60). Three freshly drawn sets from the kit generator also
score 1,000,000 each. The house baseline (plain-astar 1.0.0) scores 200,000
on the same practice set.
