#!/usr/bin/env python3
"""
Generate LIBERO episode initial states (MuJoCo flattened states) and save them as per-task `*.pruned_init` files.

Assuming this script is run from the LIBERO repository root, it writes *sets* of init states to:
  libero/libero/init_state_set/set_{i}/{libero_goal,libero_spatial,libero_object}/{task}.pruned_init

Seeding semantics:
- The RNG seed is set to 0 exactly three times total: once before generating all `libero_goal` sets,
  once before all `libero_spatial` sets, and once before all `libero_object` sets.

Each `{task}.pruned_init` matches LIBERO's `init_files/` format: a torch-saved `numpy.ndarray` with `dtype=float64`.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def _make_env(task, env_img_res: int) -> OffScreenRenderEnv:
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=env_img_res,
        camera_widths=env_img_res,
    )


def _default_out_root() -> Path:
    # Relative to the LIBERO repo root (i.e., where this script is intended to be run from)
    return Path("libero") / "libero" / "init_state_set"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LIBERO init-state sets in init_state_set/ format.")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--env_img_res", type=int, default=256)
    p.add_argument(
        "--num_sets",
        type=int,
        default=5,
        help="Number of init-state sets to generate (creates set_0 ... set_{num_sets-1}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used before generating each suite (set exactly once per suite).",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Output root directory (default: LIBERO/libero/libero/init_state_set).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    libero_root = Path.cwd()
    out_root = Path(args.out_root) if args.out_root else _default_out_root()
    if not out_root.is_absolute():
        out_root = libero_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    suite_names = ["libero_goal", "libero_spatial", "libero_object"]
    benchmark_dict = benchmark.get_benchmark_dict()

    # Pre-instantiate suites to compute total number of init states
    task_suites = {}
    total_states = 0
    for suite_name in suite_names:
        task_suite = benchmark_dict[suite_name]()
        task_suites[suite_name] = task_suite
        total_states += task_suite.n_tasks * args.num_sets * args.num_trials_per_task

    progress = tqdm(total=total_states, desc="Generating init states", unit="state")

    for suite_name in suite_names:
        _set_seed(args.seed)

        task_suite = task_suites[suite_name]
        for set_idx in range(args.num_sets):
            suite_dir = out_root / f"set_{set_idx}" / suite_name
            suite_dir.mkdir(parents=True, exist_ok=True)

            for task_id in range(task_suite.n_tasks):
                task = task_suite.get_task(task_id)
                env = _make_env(task, args.env_img_res)

                states = []
                for _ in range(args.num_trials_per_task):
                    env.reset()
                    while env.check_success():
                        env.reset()
                    states.append(env.get_sim_state())
                    progress.update(1)
                env.close()

                states_np = np.stack(states, axis=0).astype(np.float64)
                torch.save(states_np, suite_dir / f"{task.name}.pruned_init")

            print(f"Wrote {suite_name} set_{set_idx} to {suite_dir}")

    progress.close()


if __name__ == "__main__":
    main()
