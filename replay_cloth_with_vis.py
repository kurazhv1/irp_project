#!/usr/bin/env python3
"""
Replay cloth manipulation actions with LIVE VISUALIZATION using original TableClothSimEnvironment.

This script uses the REAL cloth physics simulation from the original codebase with show_vis=True.

WARNING: This may cause segfault during env.step() - it's a known mujoco-py + cloth viewer issue.
         If it crashes, use replay_cloth_full.py in headless mode instead.

Usage:
    conda activate irp_legacy
    python replay_cloth_with_vis.py <action_log.json>

Example:
    python replay_cloth_with_vis.py outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def replay_with_visualization(action_log_path):
    """
    Replay actions from action log with LIVE visualization.
    
    Args:
        action_log_path: Path to action log JSON file
    """
    print("=" * 70)
    print("🎬 IRP CLOTH MANIPULATION REPLAY WITH LIVE VISUALIZATION")
    print("=" * 70)
    
    # Load action log
    with open(action_log_path, 'r') as f:
        action_data = json.load(f)
    
    metadata = action_data['metadata']
    actions = action_data['actions']
    
    print(f"\n📋 Episode Info:")
    print(f"   Run ID:        {metadata['run_id']}")
    print(f"   Rope ID:       {metadata['rope_id']}")
    print(f"   Rope params:   {metadata['rope_param']}")
    print(f"   Goal alpha:    {metadata['goal_alpha']}")
    print(f"   Total steps:   {len(actions)}")
    print(f"   Init action:   {metadata['init_action']}")
    
    # Create environment with show_vis=True
    print(f"\n🔧 Creating environment with visualization...")
    rope_param = metadata['rope_param']
    
    # Rope config (from eval_irp_cloth_sim.yaml)
    rope_config = dict(
        table_height=0.8,
        table_y=1,
        table_size=1.2,
        cloth_spacing=rope_param[0] / 12,  # Varies per rope
        cloth_density=rope_param[1]         # Varies per rope
    )
    
    # Controller config (from eval_irp_cloth_sim.yaml)
    controller_config = dict(
        joint_names=['gy', 'gz'],
        kp=100000,
        kv=100000
    )
    
    print(f"   Creating TableClothSimEnvironment(show_vis=True)...")
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        show_vis=True,  # Enable visualization!
        obs_topdown=False  # From config
    )
    print(f"   ✓ Environment initialized with viewer")
    
    # Set up goal and loss function (same as eval script)
    print(f"\n🎯 Setting up goal and loss function...")
    goal = env.get_cloth_goal(metadata['goal_alpha'])
    loss_func = env.get_traj_loss_func(goal, n_step=len(actions), measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    print(f"   ✓ Loss function configured (goal_alpha={metadata['goal_alpha']})")
    
    print(f"\n⚠️  WARNING: Visualization may cause segfault during env.step()")
    print(f"    This is a known mujoco-py + cloth physics issue.")
    print(f"    If it crashes, use replay_cloth_full.py (headless) instead.\n")
    
    input("Press ENTER to start replay with visualization...")
    
    # Replay each step
    print(f"\n▶️  Starting replay...\n")
    print("=" * 70)
    
    successful_steps = 0
    total_loss_diff = 0.0
    
    for action_step in actions:
        step_id = action_step['step_id']
        action = np.array(action_step['action'])  # Full 4D action [x, y, z, theta]
        logged_loss = action_step['loss']
        
        print(f"\nStep {step_id + 1}/{len(actions)}:")
        print(f"  Action: {action}")
        print(f"  Logged loss: {logged_loss:.6f}")
        
        try:
            # Take action in environment (this may segfault)
            obs, loss, done, info = env.step(action)
            
            loss_diff = abs(loss - logged_loss)
            total_loss_diff += loss_diff
            successful_steps += 1
            
            print(f"  → Replayed loss: {loss:.6f}")
            print(f"  → Difference: {loss_diff:.6f}")
            
            if loss_diff > 0.01:
                print(f"  ⚠️  Large difference detected!")
            
        except Exception as e:
            print(f"  ❌ Error during step: {type(e).__name__}: {e}")
            print(f"     This is expected - cloth simulation can be unstable")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 REPLAY SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully completed {successful_steps}/{len(actions)} steps")
    
    if successful_steps > 0:
        mean_diff = total_loss_diff / successful_steps
        print(f"📈 Loss Statistics:")
        print(f"   Mean difference from logged: {mean_diff:.6f}")
    
    print("\n💡 Viewer should still be open. Close window to exit.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Replay cloth manipulation with LIVE visualization'
    )
    parser.add_argument(
        'action_log',
        type=str,
        help='Path to action log JSON file'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.action_log):
        print(f"❌ Error: Action log not found: {args.action_log}")
        sys.exit(1)
    
    print(f"📂 Loading: {args.action_log}\n")
    
    try:
        replay_with_visualization(args.action_log)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
