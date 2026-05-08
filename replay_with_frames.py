#!/usr/bin/env python3
"""
Replay with frame capture (no live visualization to avoid segfault)
Saves frames as images that can be converted to video
"""

import json
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import sys
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import mujoco_py as mj

def load_action_log(log_path):
    """Load action log from JSON file"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data

def replay_with_frames(action_log, output_dir):
    """
    Replay and save frames using offscreen rendering
    """
    metadata = action_log['metadata']
    actions = action_log['actions']
    
    print("\n=== Replay with Frame Capture ===")
    print(f"Total steps: {len(actions)}")
    print(f"Rope params: {metadata['rope_param']}")
    
    # Setup environment WITHOUT viewer
    rope_config = {
        'table_height': 0.8,
        'table_y': 1,
        'table_size': 1.2,
        'cloth_spacing': metadata['rope_param'][0] / 12,
        'cloth_density': metadata['rope_param'][1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print("\nCreating environment (no live viewer)...")
    env = TableClothSimEnvironment(
        rope_config, 
        controller_config,
        obs_topdown=False,
        show_vis=False  # No live visualization
    )
    
    # Create offscreen renderer
    print("Setting up offscreen renderer...")
    width, height = 640, 480
    
    # Get sim from environment
    sim = env.sim
    
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup goal
    goal = env.get_cloth_goal(metadata['goal_alpha'])
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    print(f"\nCapturing frames to: {output_path}")
    print("Replaying and rendering...\n")
    
    frames = []
    
    # Render initial state
    try:
        sim.render(width, height, camera_name="fixed")
        frame = sim.render(width, height, camera_name="fixed", mode='offscreen')
        frame = frame[::-1, :, :]  # Flip vertically
        frames.append(frame)
        cv2.imwrite(str(output_path / f"frame_init.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("✓ Rendered initial frame")
    except Exception as e:
        print(f"Warning: Could not render initial frame: {e}")
    
    # Replay actions
    for i, action_step in enumerate(tqdm(actions, desc="Replaying")):
        action = np.array(action_step['action'])
        
        try:
            # Step environment
            observation, loss, _, info = env.step(action)
            
            # Try to render frame
            try:
                frame = sim.render(width, height, camera_name="fixed", mode='offscreen')
                frame = frame[::-1, :, :]  # Flip vertically
                frames.append(frame)
                
                # Save frame
                frame_path = output_path / f"frame_{i:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"\nWarning: Could not render frame {i}: {e}")
                
        except Exception as e:
            print(f"\nError at step {i}: {e}")
            break
    
    print(f"\n✓ Saved {len(frames)} frames to {output_path}")
    
    # Create video if ffmpeg available
    if len(frames) > 0:
        video_path = output_path.parent / f"{output_path.name}_replay.mp4"
        print(f"\nCreating video: {video_path}")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f"✓ Video saved: {video_path}")
        except Exception as e:
            print(f"Could not create video: {e}")
            print("You can create video manually with:")
            print(f"  ffmpeg -framerate 10 -i {output_path}/frame_%04d.png -c:v libx264 {video_path}")
    
    return frames

def main():
    parser = argparse.ArgumentParser(description="Replay with frame capture")
    parser.add_argument("log_file", type=str, help="Path to action_log JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for frames")
    
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Action log not found: {log_path}")
    
    print(f"Loading: {log_path}")
    action_log = load_action_log(log_path)
    
    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = log_path.parent / f"frames_{log_path.stem}"
    
    # Replay
    try:
        frames = replay_with_frames(action_log, output_dir)
        
        if frames:
            print("\n✓ Replay completed!")
            print(f"\nFrames saved to: {output_dir}")
            print(f"Total frames: {len(frames)}")
        else:
            print("\n✗ No frames captured")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
