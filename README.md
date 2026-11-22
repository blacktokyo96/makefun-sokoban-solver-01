# Room-decomposed push-optimal Sokoban solver

For make.fun's [Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121) bounty. Cut the hallway out, solve each room push-optimally with A* (reverse-pull liveness, frozen-2x2 pruning, min-assignment heuristic), add the rooms up. Deterministic, ~1 s per set.

Run: `MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> node main.ts`
