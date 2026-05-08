#!/usr/bin/env python3
"""
Full cloth simulation replay using action logs from eval_irp_cloth_sim.py
Uses the REAL TableClothSimEnvironment with cloth physics (mujoco-py)
"""

import json
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def load_action_log(log_path):
    """Load action log JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def replay_with_cloth_physics(action_log, show_vis=False, save_frames=False):
    """
    Replay actions with FULL cloth physics simulation
    
    Args:
        action_log: Action log dictionary
        show_vis: Show visualization (WARNING: may cause segfault)
        save_frames: Save rendered frames
    """
    metadata = action_log['metadata']
    actions = action_log['actions']
    
    print("\n" + "="*70)
    print("🧵 CLOTH PHYSICS SIMULATION REPLAY")
    print("="*70)
    print(f"\n📋 Episode Metadata:")
    print(f"   Run ID:        {metadata['run_id']}")
    print(f"   Rope ID:       {metadata['rope_id']}")
    print(f"   Rope params:   {metadata['rope_param']}")
    print(f"   Goal alpha:    {metadata['goal_alpha']}")
    print(f"   Total steps:   {len(actions)}")
    print(f"   Init action:   {metadata['init_action']}")
    
    # Create environment with EXACT same parameters as eval
    rope_param = metadata['rope_param']
    rope_config = {
        'table_height': 0.8,
        'table_y': 1,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,  # Same as eval
        'cloth_density': rope_param[1],       # Same as eval
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print(f"\n🔧 Environment Configuration:")
    print(f"   Cloth spacing: {rope_config['cloth_spacing']:.6f}")
    print(f"   Cloth density: {rope_config['cloth_density']:.4f}")
    print(f"   Table height:  {rope_config['table_height']}")
    print(f"   Visualization: {'ON (may segfault!)' if show_vis else 'OFF (safe)'}")
    
    print(f"\n⚙️  Creating TableClothSimEnvironment...")
    
    try:
        env = TableClothSimEnvironment(
            rope_config, 
            controller_config,
            obs_topdown=False,
            show_vis=show_vis
        )
        print(f"   ✓ Environment created successfully!")
        print(f"   ✓ Cloth physics loaded (mujoco-py)")
        
    except Exception as e:
        print(f"   ✗ ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup goal (same as eval)
    print(f"\n🎯 Setting up goal...")
    goal_alpha = metadata['goal_alpha']
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    print(f"   ✓ Goal configured (alpha={goal_alpha})")
    
    # Reset environment
    print(f"\n🔄 Resetting environment...")
    obs = env.reset()
    print(f"   ✓ Reset complete")
    print(f"   ✓ Observation shape: {obs.shape}")
    
    print(f"\n" + "="*70)
    print(f"▶️  STARTING CLOTH SIMULATION")
    print(f"="*70 + "\n")
    
    frames = []
    losses_replayed = []
    losses_logged = []
    
    # Replay each action
    for i, action_data in enumerate(tqdm(actions, desc="Simulating cloth")):
        action = np.array(action_data['action'])
        logged_loss = action_data['loss']
        
        try:
            # Execute action in environment (FULL CLOTH PHYSICS!)
            obs, replayed_loss, done, info = env.step(action)
            
            losses_replayed.append(replayed_loss)
            losses_logged.append(logged_loss)
            
            # Save frame if requested
            if save_frames:
                frames.append(obs.copy())
            
            # Print progress
            diff = abs(replayed_loss - logged_loss)
            status = "✓" if diff < 0.1 else "⚠" if diff < 0.5 else "✗"
            
            print(f"{status} Step {i+1:2d}/{len(actions)}: "
                  f"action=[{action[0]:.3f}, {action[1]:.3f}], "
                  f"logged={logged_loss:.4f}, "
                  f"replayed={replayed_loss:.4f}, "
                  f"diff={diff:.4f}")
            
        except Exception as e:
            print(f"\n⚠️  Warning at step {i+1}: {e}")
            # Continue despite errors (cloth physics can be unstable)
            continue
    
    print(f"\n" + "="*70)
    print(f"✅ SIMULATION COMPLETE")
    print(f"="*70)
    
    # Statistics
    losses_replayed = np.array(losses_replayed)
    losses_logged = np.array(losses_logged)
    diffs = np.abs(losses_replayed - losses_logged)
    
    print(f"\n📊 Statistics:")
    print(f"   Steps completed:    {len(losses_replayed)}/{len(actions)}")
    print(f"   Logged loss:        min={losses_logged.min():.4f}, "
          f"max={losses_logged.max():.4f}, "
          f"mean={losses_logged.mean():.4f}")
    print(f"   Replayed loss:      min={losses_replayed.min():.4f}, "
          f"max={losses_replayed.max():.4f}, "
          f"mean={losses_replayed.mean():.4f}")
    print(f"   Difference:         min={diffs.min():.4f}, "
          f"max={diffs.max():.4f}, "
          f"mean={diffs.mean():.4f}")
    
    if save_frames and frames:
        print(f"\n💾 Frames: {len(frames)} captured")
        print(f"   Frame shape: {frames[0].shape}")
    
    return {
        'losses_replayed': losses_replayed,
        'losses_logged': losses_logged,
        'diffs': diffs,
        'frames': frames if save_frames else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay IRP action logs with FULL cloth physics simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe replay (no visualization)
  conda activate irp_legacy
  python simulate_cloth_replay.py outputs/*/action_logs/action_log_*.json
  
  # With visualization (WARNING: may segfault!)
  python simulate_cloth_replay.py action_log.json --viz
  
  # Save frames
  python simulate_cloth_replay.py action_log.json --save-frames
        """
    )
    
    parser.add_argument(
        "action_log",
        type=str,
        help="Path to action_log JSON file from eval_irp_cloth_sim.py"
    )
    
    parser.add_argument(
        "--viz", "--show-vis",
        action="store_true",
        help="Show live visualization (WARNING: may cause segfault!)"
    )
    
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save rendered frames"
    )
    
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.action_log)
    if not log_path.exists():
        print(f"ERROR: File not found: {log_path}")
        return 1
    
    print(f"📂 Loading: {log_path}")
    action_log = load_action_log(log_path)
    
    # Replay
    try:
        result = replay_with_cloth_physics(
            action_log, 
            show_vis=args.viz,
            save_frames=args.save_frames
        )
        
        if result:
            print(f"\n🎉 Success!")
            return 0
        else:
            print(f"\n❌ Failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
