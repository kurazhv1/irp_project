#!/usr/bin/env python
"""
Cloth simulation replay using ORIGINAL mujoco-py environment
Saves frames to video instead of live viewer (no segfault!)
"""

import sys
import json
import pathlib
import numpy as np
import mujoco_py as mj
from scipy.interpolate import CubicSpline
import imageio

# Import original environment components
from environments.table_cloth_sim_environment import (
    TableClothSimEnvironment, 
    get_cubic_control
)
from abr_control_mod.mujoco_utils import get_body_center_of_mass


def replay_episode_to_video(action_log_path: str, output_video: str = None):
    """
    Replay episode and save to video file
    
    Args:
        action_log_path: Path to action log JSON
        output_video: Output video path (default: same name as log with .mp4)
    """
    
    # Load action log
    with open(action_log_path, 'r') as f:
        log_data = json.load(f)
    
    episode_id = log_data['episode_id']
    actions = log_data['actions']
    goal_coords = np.array(log_data['goal_coords'])
    metadata = log_data['metadata']
    logged_loss = log_data['loss']
    
    print(f"📄 Loaded action log: {pathlib.Path(action_log_path).name}")
    print(f"   Episode: {episode_id}")
    print(f"   Loss: {logged_loss:.6f}")
    print(f"   Actions: {len(actions)}")
    
    # Create environment (headless, no viewer)
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
    
    print(f"🏗️  Creating environment...")
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        dt=0.01,
        max_steps=400,
        show_vis=False  # NO VIEWER = NO SEGFAULT!
    )
    
    # Setup offscreen renderer
    print(f"📹 Setting up offscreen renderer...")
    width, height = 1280, 720
    sim = env.sim
    
    # Create offscreen context (use GPU 0)
    try:
        offscreen = mj.MjRenderContextOffscreen(sim, device_id=0)
    except RuntimeError:
        print("⚠️  GPU rendering failed, trying CPU mode...")
        import os
        os.environ['MUJOCO_GL'] = 'osmesa'
        offscreen = mj.MjRenderContextOffscreen(sim, device_id=-1)
    offscreen.vopt.geomgroup[0] = 1  # Show visual geometries
    
    # Setup camera
    camera_id = sim.model.camera_name2id('fixed')
    
    # Prepare video output
    if output_video is None:
        output_video = pathlib.Path(action_log_path).with_suffix('.mp4')
    output_video = str(output_video)
    
    print(f"🎬 Recording to: {output_video}")
    print(f"   Episode: {episode_id}")
    print(f"   Actions: {len(actions)}")
    
    frames = []
    
    # Replay each action
    for action_idx, action in enumerate(actions):
        print(f"\n🎯 Action {action_idx + 1}/{len(actions)}")
        
        # Reset to initial state
        env.ctrl._load_state(*env.init_state)
        
        # Generate trajectory
        raw_action = env.action_mapper(np.array(action))
        duration, gy1, gz1, gy2 = raw_action
        gz2 = 0.05
        
        t_in = np.linspace(0, duration, 3)
        q_in = np.array([
            [0, 0],
            [gy1, gz1],
            [gy2, gz2]
        ])
        qs, dqs, ts = get_cubic_control(t_in, q_in, env.dt)
        
        # Execute action
        pad_steps = int(0.2 / env.dt)
        n_steps = min(env.max_steps, len(qs) + pad_steps + 20)
        
        print(f"   Duration: {duration:.2f}s")
        print(f"   Trajectory: [0,0] → [{gy1:.3f}, {gz1:.3f}] → [{gy2:.3f}, {gz2:.3f}]")
        print(f"   Steps: {n_steps}")
        
        hist = []
        
        # Render every Nth frame to save time
        render_every = 5  # Render every 5 steps (50 FPS)
        
        for i in range(n_steps):
            # Control
            ii = max(min(i, len(qs) - 1), 0)
            q = qs[ii]
            dq = dqs[ii]
            u = env.ctrl.generate(q, dq)
            env.ctrl.send_forces(u)
            sim.step()
            
            # Record keypoints
            kp_com = get_body_center_of_mass(sim.data, env.kp_ids)
            hist.append(kp_com)
            
            # Render frame
            if i % render_every == 0:
                offscreen.render(width, height, camera_id)
                rgb = offscreen.read_pixels(width, height, depth=False)
                rgb = rgb[::-1, :, :]  # Flip vertically
                frames.append(rgb)
        
        hist = np.array(hist)
        
        # Compute loss
        final_cloth_pos = hist[-1]
        diff = final_cloth_pos - goal_coords
        dists = np.linalg.norm(diff, axis=-1)
        loss = np.mean(dists)
        
        print(f"   Final loss: {loss:.6f}")
        print(f"   Mean distance to goal: {loss * 100:.1f}cm")
        print(f"   Frames captured: {len(frames)}")
    
    # Save video
    print(f"\n💾 Saving video...")
    print(f"   Total frames: {len(frames)}")
    print(f"   Output: {output_video}")
    
    fps = 10  # 10 FPS for smooth playback
    imageio.mimsave(output_video, frames, fps=fps, quality=8, macro_block_size=1)
    
    print(f"✅ Video saved successfully!")
    print(f"   Path: {output_video}")
    print(f"   Duration: {len(frames) / fps:.1f}s")
    
    return output_video


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_cloth_mujocopy.py <action_log.json> [output.mp4]")
        sys.exit(1)
    
    action_log_path = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        replay_episode_to_video(action_log_path, output_video)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
