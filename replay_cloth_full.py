#!/usr/bin/env python3
"""
Full cloth simulation replay using TableClothSimEnvironment
This script replays action logs with REAL cloth physics simulation
"""

import json
import pathlib
import argparse
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mujoco_py as mj
from mujoco_py import GlfwContext

# CRITICAL FIX: Initialize GLFW context for offscreen rendering
# Without this, visualization will segfault
# See: https://github.com/openai/mujoco-py/issues/390
GlfwContext(offscreen=True)

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def load_action_log(log_path):
    """Load action log from JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def create_environment_from_metadata(metadata, show_vis=False):
    """
    Create TableClothSimEnvironment from action log metadata
    This uses the REAL cloth environment with full physics
    """
    rope_param = metadata['rope_param']
    
    # Exact same config as in eval_irp_cloth_sim.py
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,  # Convert to spacing
        'cloth_density': rope_param[1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print(f"\n🔧 Creating TableClothSimEnvironment:")
    print(f"   Cloth spacing: {rope_config['cloth_spacing']:.4f}")
    print(f"   Cloth density: {rope_config['cloth_density']:.2f}")
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        obs_topdown=False,
        show_vis=show_vis  # WARNING: show_vis=True may cause segfault!
    )
    
    # Set up goal (same as eval)
    goal_alpha = metadata.get('goal_alpha', 0.0)
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    
    print(f"   ✓ Environment created with REAL cloth physics")
    print(f"   Goal alpha: {goal_alpha}")
    
    return env


def replay_with_full_cloth_sim(action_log, show_vis=False, save_images=False, save_states=None):
    """
    Replay actions with FULL cloth simulation
    
    Args:
        action_log: Loaded action log dictionary
        show_vis: Show visualization (WARNING: may segfault!)
        save_images: Save observation images to disk
        save_states: Path to save qpos states (.npz file) for later visualization
    """
    metadata = action_log['metadata']
    actions = action_log['actions']
    
    print("\n" + "="*70)
    print("🧵 FULL CLOTH SIMULATION REPLAY")
    print("="*70)
    print(f"\n📋 Episode Info:")
    print(f"   Run ID:        {metadata['run_id']}")
    print(f"   Rope ID:       {metadata['rope_id']}")
    print(f"   Rope params:   {metadata['rope_param']}")
    print(f"   Goal alpha:    {metadata['goal_alpha']}")
    print(f"   Total steps:   {len(actions)}")
    
    if show_vis:
        print("\n⚠️  WARNING: Visualization enabled - may cause segfault!")
        print("   If it crashes, run with default (show_vis=False)")
    
    # Create environment with REAL cloth
    env = create_environment_from_metadata(metadata, show_vis=show_vis)
    
    # Get initial observation (no reset needed - environment starts fresh)
    print(f"\n🔄 Getting initial state...")
    # Note: TableClothSimEnvironment doesn't have reset(), 
    # it starts in initial state upon creation
    
    # Optional: prepare image saving
    if save_images:
        import cv2
        output_dir = pathlib.Path("cloth_replay_images")
        output_dir.mkdir(exist_ok=True)
        print(f"   💾 Will save images to: {output_dir}")
    
    print(f"\n▶️  Starting cloth simulation...")
    print("="*70 + "\n")
    
    # Replay each action
    losses = []
    saved_states = [] if save_states else None
    
    # Save initial state
    if save_states:
        saved_states.append(env.sim.data.qpos.copy())
    
    for i, action in enumerate(tqdm(actions, desc="🧵 Simulating cloth")):
        action = np.array(action)  # Actions are already arrays in new format
        logged_loss = None  # Loss not stored per-action in new format
        
        try:
            # Execute action in FULL cloth simulation
            obs, loss, done, info = env.step(action)
            
            losses.append(loss)
            
            # Save state after action
            if save_states:
                saved_states.append(env.sim.data.qpos.copy())
            
            # Print progress
            tqdm.write(f"   Step {i+1:2d}/{len(actions)}: Loss = {loss:.6f} ({loss*100:.2f}cm)")
            
            # Save observation image
            if save_images and (i + 1) % 4 == 0:  # Save every 4th step
                import cv2
                cv2.imwrite(
                    str(output_dir / f"step_{i+1:03d}.png"), 
                    cv2.COLOR_RGB2BGR(obs)
                )
            
        except mj.MujocoException as e:
            tqdm.write(f"\n⚠️  MuJoCo error at step {i+1}: {e}")
            tqdm.write(f"   (This is normal - cloth simulation can be unstable)")
            continue
        except Exception as e:
            tqdm.write(f"\n❌ Error at step {i+1}: {e}")
            break
    
    # Save states to file
    if save_states and saved_states:
        print(f"\n💾 Saving {len(saved_states)} states to: {save_states}")
        np.savez_compressed(
            save_states,
            states=np.array(saved_states),
            metadata={
                'episode_id': action_log.get('episode_id', 'unknown'),
                'rope_id': metadata['rope_id'],
                'loss': losses[-1] if losses else 0,
                'rope_config': env.rope_config,
                'n_actions': len(actions),
            }
        )
        print(f"   ✅ States saved successfully")
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    if losses:
        print(f"\n✅ Successfully completed {len(losses)}/{len(actions)} steps")
        print(f"\n📈 Loss Statistics:")
        print(f"   Mean loss:     {np.mean(losses):.4f}")
        print(f"   Min loss:      {np.min(losses):.4f}")
        print(f"   Max loss:      {np.max(losses):.4f}")
        print(f"   Final loss:    {losses[-1]:.4f}")
        
        if save_images:
            print(f"\n💾 Saved {len(losses)//4 + 1} images to: {output_dir}")
    else:
        print(f"\n❌ No steps completed successfully")
    
    print("="*70)
    
    # Note about visualization
    if not show_vis:
        print("\n💡 Note: Running in headless mode (no visualization)")
        print("   To see visualization (risky!): use --show-vis flag")
    
    return losses


def main():
    parser = argparse.ArgumentParser(
        description="Replay action logs with FULL cloth simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Safe: Headless replay with full cloth physics
  conda activate irp_legacy
  python replay_cloth_full.py outputs/*/action_logs/action_log_*.json
  
  # Save observation images
  python replay_cloth_full.py action_log.json --save-images
  
  # Risky: Enable visualization (may segfault!)
  python replay_cloth_full.py action_log.json --show-vis
  
This script uses the REAL TableClothSimEnvironment with full cloth physics,
exactly as used during eval_irp_cloth_sim.py evaluation.
        """
    )
    
    parser.add_argument(
        "action_log",
        type=str,
        help="Path to action_log JSON file"
    )
    
    parser.add_argument(
        "--show-vis",
        action="store_true",
        help="Show visualization (WARNING: may cause segfault!)"
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save observation images to disk"
    )
    
    parser.add_argument(
        "--save-states",
        type=str,
        metavar="FILE",
        help="Save qpos states to .npz file for later visualization"
    )
    
    args = parser.parse_args()
    
    try:
        # Load action log
        log_path = pathlib.Path(args.action_log)
        if not log_path.exists():
            print(f"❌ ERROR: File not found: {log_path}")
            return 1
        
        print(f"📂 Loading: {log_path}")
        action_log = load_action_log(log_path)
        
        # Replay with full cloth simulation
        losses = replay_with_full_cloth_sim(
            action_log, 
            show_vis=args.show_vis,
            save_images=args.save_images,
            save_states=args.save_states
        )
        
        if losses:
            print(f"\n✅ SUCCESS!")
            return 0
        else:
            print(f"\n⚠️  No steps completed")
            return 1
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
