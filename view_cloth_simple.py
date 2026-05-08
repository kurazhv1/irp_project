#!/usr/bin/env python3
"""
Simple viewer for cloth simulation using the original evaluation data.

This script shows the ACTUAL cloth simulation visually by:
1. Running headless simulation (accurate physics)
2. Saving frames as images
3. Opening images in sequence

Usage:
    conda activate irp_legacy
    python view_cloth_simple.py <action_log.json>
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import mujoco_py as mj


def visualize_cloth_simulation(action_log_path, output_dir="cloth_frames"):
    """
    Run cloth simulation and save frames.
    """
    print("=" * 70)
    print("🎬 CLOTH SIMULATION VIEWER")
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
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_path.absolute()}")
    
    # Create environment (headless)
    print(f"\n🔧 Creating environment...")
    rope_param = metadata['rope_param']
    
    rope_config = dict(
        table_height=0.8,
        table_y=1,
        table_size=1.2,
        cloth_spacing=rope_param[0] / 12,
        cloth_density=rope_param[1]
    )
    
    controller_config = dict(
        joint_names=['gy', 'gz'],
        kp=100000,
        kv=100000
    )
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        show_vis=False,  # Headless to avoid segfault
        obs_topdown=False
    )
    print("   ✅ Environment created")
    
    # Setup loss function
    goal = env.get_cloth_goal(metadata['goal_alpha'])
    loss_func = env.get_traj_loss_func(goal, n_step=len(actions), measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    # Get observations (these are the actual images!)
    print(f"\n▶️  Running simulation and collecting frames...\n")
    
    frames = []
    successful_steps = 0
    
    for action_step in tqdm(actions, desc="Processing"):
        step_id = action_step['step_id']
        action = np.array(action_step['action'])
        logged_loss = action_step['loss']
        
        try:
            obs, loss, done, info = env.step(action)
            
            # obs shape: (9, 256, 256) - 9 channels, 256x256 image
            # Let's use first 3 channels as RGB
            if obs.shape[0] >= 3:
                # Take first 3 channels and transpose to (256, 256, 3)
                frame = obs[:3].transpose(1, 2, 0)
                # Normalize to 0-255
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
            else:
                # Grayscale
                frame = obs[0]
                frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Add text overlay
            cv2.putText(frame, f"Step {step_id + 1}/{len(actions)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Loss: {loss:.4f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame
            frame_path = output_path / f"frame_{step_id:03d}.png"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
            
            successful_steps += 1
            
        except mj.MujocoException as e:
            print(f"\n⚠️  MuJoCo exception at step {step_id + 1}: {e}")
            break
        except Exception as e:
            print(f"\n❌ Error at step {step_id + 1}: {e}")
            break
    
    print("\n" + "=" * 70)
    print(f"📊 COMPLETE")
    print("=" * 70)
    print(f"✅ Saved {len(frames)} frames")
    print(f"📁 Location: {output_path.absolute()}")
    print("\n💡 View frames:")
    print(f"   eog {output_path}/*.png")
    print(f"   feh {output_path}/*.png")
    print(f"   Or open folder in file manager")
    print("=" * 70)
    
    # Try to open first frame
    if frames:
        first_frame = frames[0]
        print(f"\n🖼️  Opening first frame: {first_frame}")
        os.system(f"xdg-open {first_frame} &")
    
    return frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description='View cloth simulation frames')
    parser.add_argument('action_log', type=str, help='Path to action log JSON')
    parser.add_argument('--output-dir', type=str, default='cloth_frames', 
                       help='Output directory for frames')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.action_log):
        print(f"❌ Error: Action log not found: {args.action_log}")
        sys.exit(1)
    
    try:
        visualize_cloth_simulation(args.action_log, args.output_dir)
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
