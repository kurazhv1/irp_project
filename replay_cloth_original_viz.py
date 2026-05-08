#!/usr/bin/env python3
"""
Replay action logs with ORIGINAL cloth simulation and LIVE VISUALIZATION.
This uses the real TableClothSimEnvironment with show_vis=True.

WARNING: This may cause segfaults on some systems. If it crashes, use:
  - replay_cloth_full.py (headless, stable)
  - replay_viewer_mj3.py (MuJoCo 3, simplified physics)
"""

import argparse
import json
import pathlib
import numpy as np
import sys
from tqdm import tqdm

# Add project root to path
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def create_environment_from_metadata(metadata: dict, show_vis: bool = True):
    """Create environment matching the logged configuration."""
    
    # Extract rope config from metadata
    rope_param = metadata['rope_params']
    
    # Build rope_config matching original
    rope_config = {
        'cloth_spacing': rope_param[0] / 12,  # rope_param[0] is cloth_size
        'cloth_density': rope_param[1],
        'table_size': 1.0,
        'table_y': 0.5,
    }
    
    # Controller config (standard from eval_irp_cloth_sim.py)
    controller_config = {
        'kp': 800,
        'kv': 160,
    }
    
    print(f"🔧 Creating ORIGINAL environment with show_vis={show_vis}...")
    print(f"   Cloth spacing: {rope_config['cloth_spacing']:.4f}")
    print(f"   Cloth density: {rope_config['cloth_density']:.2f}")
    
    try:
        env = TableClothSimEnvironment(
            rope_config=rope_config,
            controller_config=controller_config,
            show_vis=show_vis  # Enable visualization!
        )
        print("   ✓ Environment initialized with viewer")
        return env
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        raise


def replay_with_visualization(action_log_path: str):
    """Replay action log with full cloth simulation and live visualization."""
    
    # Load action log
    print(f"📂 Loading: {action_log_path}\n")
    with open(action_log_path, 'r') as f:
        action_log = json.load(f)
    
    metadata = action_log['metadata']
    steps = action_log['steps']
    
    print("=" * 70)
    print("🎬 ORIGINAL IRP CLOTH SIMULATION - LIVE VISUALIZATION")
    print("=" * 70)
    print()
    print("📋 Episode Info:")
    print(f"   Run ID:        {metadata['run_id']}")
    print(f"   Rope ID:       {metadata['rope_id']}")
    print(f"   Rope params:   {metadata['rope_params']}")
    print(f"   Goal alpha:    {metadata['goal_alpha']}")
    print(f"   Total steps:   {len(steps)}")
    print(f"   Init action:   {metadata['init_action']}")
    print()
    
    # Create environment WITH visualization
    try:
        env = create_environment_from_metadata(metadata, show_vis=True)
    except Exception as e:
        print(f"\n❌ Environment creation failed: {e}")
        print("\n💡 Try these alternatives:")
        print("   1. replay_cloth_full.py (headless, stable)")
        print("   2. replay_viewer_mj3.py (MuJoCo 3, simplified)")
        return
    
    # Get goal
    goal_alpha = metadata['goal_alpha']
    goal = env.get_cloth_goal(goal_alpha)
    
    # Setup loss function
    loss_func = env.get_traj_loss_func(
        goal=goal,
        n_sample=100,
        threshold=0.05
    )
    env.set_loss_func(loss_func)
    
    print()
    print("🎥 Viewer Controls:")
    print("   - Close window to exit")
    print("   - Viewer updates during simulation")
    print("=" * 70)
    print()
    print("▶️  Starting replay with LIVE visualization...")
    print()
    
    # Replay each step
    replayed_losses = []
    logged_losses = []
    successful_steps = 0
    
    for i, step_data in enumerate(tqdm(steps, desc="Replaying steps")):
        step_num = step_data['step']
        action = np.array(step_data['action'])
        logged_loss = step_data['loss']
        
        try:
            # Execute action (viewer will render automatically!)
            observation, info = env.step(action)
            replayed_loss = info['loss']
            
            replayed_losses.append(replayed_loss)
            logged_losses.append(logged_loss)
            successful_steps += 1
            
            # Print progress
            diff = abs(replayed_loss - logged_loss)
            tqdm.write(f"Step {step_num}/{len(steps)}: action={action}, logged_loss={logged_loss:.4f}")
            tqdm.write(f"   → replayed_loss={replayed_loss:.4f}, diff={diff:.6f}")
            
        except Exception as e:
            tqdm.write(f"\n⚠️  Step {step_num} failed: {e}")
            tqdm.write(f"   Continuing to next step...")
            continue
    
    print()
    print("=" * 70)
    
    if successful_steps > 0:
        print(f"✅ Successfully completed {successful_steps}/{len(steps)} steps")
        print()
        print("📈 Loss Statistics:")
        print(f"   Mean loss:     {np.mean(replayed_losses):.4f}")
        print(f"   Min loss:      {np.min(replayed_losses):.4f}")
        print(f"   Max loss:      {np.max(replayed_losses):.4f}")
        print(f"   Final loss:    {replayed_losses[-1]:.4f}")
        print()
        print("🔍 Comparison with logged losses:")
        diffs = np.abs(np.array(replayed_losses) - np.array(logged_losses))
        print(f"   Mean diff:     {np.mean(diffs):.4f}")
        print(f"   Max diff:      {np.max(diffs):.4f}")
    else:
        print("❌ No steps completed successfully")
    
    print("=" * 70)
    print()
    print("💡 Viewer window should still be open. Close it to exit.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay action logs with ORIGINAL cloth simulation and LIVE visualization"
    )
    parser.add_argument(
        'action_log',
        type=str,
        help='Path to action log JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    log_path = pathlib.Path(args.action_log)
    if not log_path.exists():
        print(f"❌ Error: Action log not found: {log_path}")
        return 1
    
    try:
        replay_with_visualization(str(log_path))
        return 0
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
