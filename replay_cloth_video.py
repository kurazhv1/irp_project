#!/usr/bin/env python3
"""
Replay cloth manipulation and save as video using offscreen rendering.

This avoids the segfault issue by using mujoco_py offscreen rendering
instead of the interactive viewer.

Usage:
    conda activate irp_legacy
    python replay_cloth_video.py <action_log.json> --output video.mp4

Example:
    python replay_cloth_video.py outputs/2025-10-26/16-05-10/action_logs/action_log_20251026_160517_rope0_goal0.json --output cloth_replay.mp4
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import mujoco_py as mj


def replay_and_save_video(action_log_path, output_video, fps=10, resolution=(640, 480)):
    """
    Replay actions and save as video using offscreen rendering.
    
    Args:
        action_log_path: Path to action log JSON file
        output_video: Output video file path
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    print("=" * 70)
    print("🎬 IRP CLOTH MANIPULATION - VIDEO RECORDING")
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
    
    # Create environment WITHOUT visualization (headless)
    print(f"\n🔧 Creating environment (headless with offscreen rendering)...")
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
        show_vis=False,  # NO visualization to avoid segfault
        obs_topdown=False
    )
    print(f"   ✓ Environment created")
    
    # Setup offscreen renderer
    print(f"   Setting up offscreen renderer ({resolution[0]}x{resolution[1]})...")
    camera_id = 0  # Default camera
    renderer = mj.MjRenderContextOffscreen(env.sim, device_id=-1)
    renderer.render(resolution[0], resolution[1], camera_id)
    print(f"   ✓ Offscreen renderer ready")
    
    # Setup loss function
    goal = env.get_cloth_goal(metadata['goal_alpha'])
    loss_func = env.get_traj_loss_func(goal, n_step=len(actions), measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, resolution)
    print(f"\n📹 Recording video to: {output_video}")
    print(f"   FPS: {fps}, Resolution: {resolution[0]}x{resolution[1]}")
    
    # Render and save initial frame
    renderer.render(resolution[0], resolution[1], camera_id)
    frame = renderer.read_pixels(resolution[0], resolution[1], depth=False)
    frame = frame[::-1, :, :]  # Flip vertically
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    # Replay each step and record
    print(f"\n▶️  Replaying and recording...\n")
    successful_steps = 0
    
    for action_step in tqdm(actions, desc="Recording"):
        step_id = action_step['step_id']
        action = np.array(action_step['action'])
        logged_loss = action_step['loss']
        
        try:
            # Take action
            obs, loss, done, info = env.step(action)
            successful_steps += 1
            
            # Render frame
            renderer.render(resolution[0], resolution[1], camera_id)
            frame = renderer.read_pixels(resolution[0], resolution[1], depth=False)
            frame = frame[::-1, :, :]  # Flip vertically
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add text overlay
            text = f"Step {step_id + 1}/{len(actions)} | Loss: {loss:.4f}"
            cv2.putText(frame_bgr, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame_bgr)
            
        except mj.MujocoException as e:
            print(f"\n⚠️  MuJoCo exception at step {step_id + 1}: {e}")
            break
        except Exception as e:
            print(f"\n❌ Error at step {step_id + 1}: {e}")
            break
    
    # Cleanup
    out.release()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 RECORDING COMPLETE")
    print("=" * 70)
    print(f"✅ Recorded {successful_steps}/{len(actions)} steps")
    print(f"📹 Video saved to: {output_video}")
    
    file_size = os.path.getsize(output_video) / (1024 * 1024)
    print(f"📦 File size: {file_size:.2f} MB")
    print("=" * 70)
    
    return output_video


def main():
    parser = argparse.ArgumentParser(
        description='Replay cloth manipulation and save as video'
    )
    parser.add_argument(
        'action_log',
        type=str,
        help='Path to action log JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='cloth_replay.mp4',
        help='Output video file path (default: cloth_replay.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Video FPS (default: 10)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Video width (default: 640)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Video height (default: 480)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.action_log):
        print(f"❌ Error: Action log not found: {args.action_log}")
        sys.exit(1)
    
    print(f"📂 Loading: {args.action_log}\n")
    
    try:
        output_file = replay_and_save_video(
            args.action_log,
            args.output,
            fps=args.fps,
            resolution=(args.width, args.height)
        )
        print(f"\n✅ Success! Play video with:")
        print(f"   vlc {output_file}")
        print(f"   mpv {output_file}")
        
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
