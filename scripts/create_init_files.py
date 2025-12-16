"""
Script to generate init_states files for LIBERO benchmark tasks.

This script creates .pruned_init files for each task in libero_goal, libero_object,
and libero_spatial suites. The generated files are compatible with the evaluation
code - just change the init_states_folder path to use them.

Usage:
    python scripts/create_init_files.py --num-init-states 50 --seed 0

The files will be saved to:
    /home/txs/Code/Policy_Eval_Done_Right/LIBERO/libero/libero/init_files_new/
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path, "../"))

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from robosuite.utils.errors import RandomizationError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate init_states files for LIBERO benchmark"
    )
    parser.add_argument(
        "--num-init-states",
        type=int,
        default=50,
        help="Number of initial states to generate per task (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/txs/Code/Policy_Eval_Done_Right/LIBERO/libero/libero/init_files_new",
        help="Output directory for init files",
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_goal", "libero_object", "libero_spatial"],
        help="Task suites to process (default: libero_goal libero_object libero_spatial)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Maximum retries for sampling a valid initial state (default: 100)",
    )
    return parser.parse_args()


def generate_init_states_for_task(
    bddl_file: str,
    num_init_states: int,
    base_seed: int,
    max_retries: int = 100,
) -> np.ndarray:
    """
    Generate init states for a single task.
    
    Args:
        bddl_file: Path to the BDDL file for the task
        num_init_states: Number of initial states to generate
        base_seed: Base random seed
        max_retries: Maximum retries for each initial state
        
    Returns:
        numpy array of shape (num_init_states, state_dim)
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    env = OffScreenRenderEnv(**env_args)
    init_states = []
    
    for i in range(num_init_states):
        seed = base_seed + i
        env.seed(seed)
        np.random.seed(seed)
        
        # Try to reset the environment, with retries for RandomizationError
        success = False
        for retry in range(max_retries):
            try:
                env.reset()
                success = True
                break
            except RandomizationError:
                # Randomization failed, try with a different seed
                new_seed = seed + (retry + 1) * 1000
                env.seed(new_seed)
                np.random.seed(new_seed)
                continue
            except Exception as e:
                print(f"[warning] Unexpected error during reset: {e}")
                new_seed = seed + (retry + 1) * 1000
                env.seed(new_seed)
                np.random.seed(new_seed)
                continue
        
        if not success:
            print(f"[error] Failed to generate init state {i} after {max_retries} retries")
            # Use the last successful state or skip
            continue
        
        # Get the flattened MuJoCo simulation state
        sim_state = env.sim.get_state().flatten()
        init_states.append(sim_state)
    
    env.close()
    
    if len(init_states) < num_init_states:
        print(f"[warning] Only generated {len(init_states)}/{num_init_states} init states")
    
    return np.stack(init_states, axis=0)


def main():
    args = parse_args()
    
    # Get the path to BDDL files
    bddl_folder = get_libero_path("bddl_files")
    
    print(f"[info] BDDL folder: {bddl_folder}")
    print(f"[info] Output directory: {args.output_dir}")
    print(f"[info] Number of init states per task: {args.num_init_states}")
    print(f"[info] Base seed: {args.seed}")
    print(f"[info] Suites to process: {args.suites}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_tasks = sum(len(libero_task_map[suite]) for suite in args.suites)
    processed = 0
    
    for suite_name in args.suites:
        print(f"\n{'='*60}")
        print(f"Processing suite: {suite_name}")
        print(f"{'='*60}")
        
        # Create suite subdirectory
        suite_output_dir = os.path.join(args.output_dir, suite_name)
        os.makedirs(suite_output_dir, exist_ok=True)
        
        task_names = libero_task_map[suite_name]
        
        for task_name in tqdm(task_names, desc=f"{suite_name}"):
            processed += 1
            
            # Construct BDDL file path
            bddl_file = os.path.join(bddl_folder, suite_name, f"{task_name}.bddl")
            
            if not os.path.exists(bddl_file):
                print(f"[error] BDDL file not found: {bddl_file}")
                continue
            
            # Output file path (using .pruned_init extension)
            output_file = os.path.join(suite_output_dir, f"{task_name}.pruned_init")
            
            # Skip if already exists
            if os.path.exists(output_file):
                print(f"[skip] {task_name} (already exists)")
                continue
            
            try:
                # Generate init states
                init_states = generate_init_states_for_task(
                    bddl_file=bddl_file,
                    num_init_states=args.num_init_states,
                    base_seed=args.seed,
                    max_retries=args.max_retries,
                )
                
                # Convert to torch tensor and save
                init_states_tensor = torch.from_numpy(init_states).float()
                torch.save(init_states_tensor, output_file)
                
                print(f"[done] {task_name}: shape={init_states_tensor.shape}")
                
            except Exception as e:
                print(f"[error] Failed to process {task_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print(f"[info] Finished! Processed {processed}/{total_tasks} tasks")
    print(f"[info] Init files saved to: {args.output_dir}")
    print()
    print("To use these init files in evaluation, update the init_states_folder path:")
    print(f'  cfg.init_states_folder = "{args.output_dir}"')
    print()


if __name__ == "__main__":
    main()

