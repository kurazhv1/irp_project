#!/usr/bin/env python3
"""
Record cloth replay to video using REAL TableClothSimEnvironment
Uses virtual display (Xvfb) to avoid segfault
"""

import json
import pathlib
import numpy as np
import sys
import cv2
import subprocess
import time
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mujoco_py as mj
from environments.table_cloth_sim_environment import TableClothSimEnvironment


def load_action_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)


def record_replay_with_viewer(action_log_path, output_video='cloth_replay_real.mp4', fps=30):
    """
    Record cloth replay using REAL physics with mujoco-py viewer
    Captures frames from the viewer to create video
    """
    action_log = load_action_log(action_log_path)
    metadata = action_log['metadata']
    actions = action_log['actions']
    
    print("=" * 70)
    print("🎬 RECORDING REAL CLOTH SIMULATION")
    print("=" * 70)
    print(f"\n📋 Episode: {action_log['episode_id']}")
    print(f"   Actions: {len(actions)}")
    print(f"   Output: {output_video}")
    
    # Create environment WITH viewer (in virtual display)
    rope_param = metadata['rope_param']
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,
        'cloth_density': rope_param[1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print(f"\n🔧 Creating environment with REAL physics...")
    print(f"   (Using mujoco-py + viewer in virtual display)")
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        obs_topdown=False,
        show_vis=True  # Enable viewer!
    )
    
    # Set up goal
    goal_alpha = metadata.get('goal_alpha', 0.0)
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    
    print(f"   ✅ Environment created")
    
    # Video writer
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"\n🎥 Recording...")
    
    frame_count = 0
    losses = []
    
    for i, action in enumerate(tqdm(actions, desc="Recording")):
        action = np.array(action)
        
        try:
            # Execute action in REAL environment
            obs, loss, done, info = env.step(action)
            losses.append(loss)
            
            # Render frames
            # Each action is ~1-2 seconds, capture multiple frames
            raw_action = info['raw_action']
            duration = raw_action[0]
            num_frames = max(int(duration * fps), fps // 2)  # At least 0.5s worth
            
            for _ in range(num_frames):
                # Get frame from viewer (if available)
                if env.viewer is not None:
                    try:
                        # Try to read pixels from viewer
                        frame = env.sim.render(width, height, mode='offscreen')
                        frame_bgr = cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                        frame_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Render error: {e}")
                        break
                
                env.viewer.render() if env.viewer else None
            
        except Exception as e:
            print(f"   ⚠️  Error at action {i+1}: {e}")
            continue
    
    video_writer.release()
    
    print(f"\n✅ Recording complete!")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {frame_count/fps:.1f}s")
    print(f"   Final loss: {losses[-1]:.4f} ({losses[-1]*100:.2f}cm)")
    print(f"   Video: {output_video}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action_log', type=str)
    parser.add_argument('--output', '-o', type=str, default='cloth_replay_real.mp4')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    
    record_replay_with_viewer(args.action_log, args.output, args.fps)
