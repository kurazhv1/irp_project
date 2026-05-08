#!/usr/bin/env python3
"""
Replay actions from action_log.json files using MuJoCo 3
This script reads saved action sequences and replays them in the simulator
"""

import json
import pathlib
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm
import time

def load_action_log(log_path):
    """Load action log from JSON file"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data

def setup_mujoco3_env(metadata):
    """
    Setup MuJoCo 3 environment based on metadata
    Note: This needs to be adapted based on your specific cloth model
    """
    # TODO: Load the appropriate XML model based on metadata
    # For now, this is a placeholder
    model_path = "assets/mujoco/cloth/cloth.xml"  # Adjust path as needed
    
    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    return model, data

def replay_actions(model, data, actions, visualize=True, save_frames=False):
    """
    Replay action sequence in MuJoCo 3
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        actions: List of action dictionaries
        visualize: Whether to show live visualization
        save_frames: Whether to save frames to disk
    """
    frames = []
    
    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for action_step in tqdm(actions, desc="Replaying actions"):
                action = np.array(action_step['action'])
                
                # Apply action (this needs to be adapted to your control scheme)
                # For example, if actions are joint positions:
                # data.ctrl[:] = action
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Save frame if requested
                if save_frames:
                    # Render frame
                    renderer = mujoco.Renderer(model, height=480, width=640)
                    renderer.update_scene(data)
                    frame = renderer.render()
                    frames.append(frame)
                
                # Control playback speed
                time.sleep(0.01)
    else:
        # Headless replay
        for action_step in tqdm(actions, desc="Replaying actions"):
            action = np.array(action_step['action'])
            
            # Apply action
            # data.ctrl[:] = action
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            if save_frames:
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data)
                frame = renderer.render()
                frames.append(frame)
    
    return frames

def save_video(frames, output_path, fps=30):
    """Save frames as video using opencv"""
    import cv2
    
    if not frames:
        print("No frames to save")
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Replay action logs in MuJoCo 3")
    parser.add_argument("log_file", type=str, help="Path to action_log JSON file")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--save-video", type=str, help="Save replay as video to specified path")
    parser.add_argument("--model", type=str, help="Path to MuJoCo XML model (overrides metadata)")
    
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Action log not found: {log_path}")
    
    print(f"Loading action log: {log_path}")
    action_log = load_action_log(log_path)
    
    # Print metadata
    metadata = action_log['metadata']
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    actions = action_log['actions']
    print(f"\nTotal steps: {len(actions)}")
    
    # Setup MuJoCo environment
    print("\nSetting up MuJoCo 3 environment...")
    try:
        if args.model:
            model = mujoco.MjModel.from_xml_path(args.model)
            data = mujoco.MjData(model)
        else:
            model, data = setup_mujoco3_env(metadata)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        print("\nPlease specify model with --model flag")
        return
    
    # Replay actions
    visualize = not args.no_vis
    save_frames = args.save_video is not None
    
    print(f"\nReplaying actions (visualize={visualize}, save_frames={save_frames})...")
    frames = replay_actions(model, data, actions, visualize=visualize, save_frames=save_frames)
    
    # Save video if requested
    if args.save_video and frames:
        video_path = pathlib.Path(args.save_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(frames, video_path)
    
    print("\nReplay complete!")

if __name__ == "__main__":
    main()
