#!/usr/bin/env python3
"""
Cloth replay viewer using PRE-RECORDED states from replay_cloth_full.py
This avoids physics differences - just plays back correct simulation results.

Usage:
1. First run: python replay_cloth_full.py <action_log.json> --save-states states.npz
2. Then run: python replay_cloth_states_viewer.py states.npz
"""

import sys
import numpy as np
from pathlib import Path
import glfw
import time
import mujoco
from jinja2 import Template


def load_states(states_path):
    """Load pre-recorded states from npz file"""
    data = np.load(states_path, allow_pickle=True)
    states = data['states']
    metadata = data['metadata'].item()
    return states, metadata


def create_rendering_model(rope_config):
    """Create MuJoCo model for rendering only"""
    template_path = Path('assets/mujoco/cloth/table_cloth_template.xml.jinja2')
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    xml_string = template.render(rope_config=rope_config)
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    return model, data


def visualize_states(model, data, states, metadata):
    """Interactive visualization of pre-recorded states"""
    print("\n🎥 Starting interactive viewer...")
    
    # Initialize GLFW
    if not glfw.init():
        print("❌ Failed to initialize GLFW")
        sys.exit(1)
    
    # Create window
    width, height = 1920, 1080
    window = glfw.create_window(width, height, "🎬 Cloth Replay (Pre-recorded States)", None, None)
    if not window:
        glfw.terminate()
        print("❌ Failed to create GLFW window")
        sys.exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Create MuJoCo rendering context
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 180
    cam.elevation = -25
    cam.distance = 1.8
    cam.lookat[:] = [0.7, 0.0, 0.1]
    
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Playback state
    current_frame = 0
    playing = False  # Start paused
    playback_speed = 1.0
    last_time = time.time()
    
    print(f"   Total frames: {len(states)}")
    print("\n💡 Controls:")
    print("   Space:      Pause/Play")
    print("   Left/Right: Step backward/forward (5 frames)")
    print("   Shift+Left/Right: Single frame step")
    print("   Up/Down:    Speed up/down")
    print("   Home:       Jump to start")
    print("   End:        Jump to end")
    print("   R:          Reset camera")
    print("   ESC:        Exit")
    print("=" * 70)
    print("\n▶️  Press SPACE to start playback")
    print("=" * 70)
    
    # Main render loop
    while not glfw.window_should_close(window):
        # Handle input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            playing = not playing
            time.sleep(0.2)
        
        shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            step = 1 if shift_pressed else 5
            current_frame = max(0, current_frame - step)
            playing = False
            time.sleep(0.05 if shift_pressed else 0.1)
        
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            step = 1 if shift_pressed else 5
            current_frame = min(len(states) - 1, current_frame + step)
            playing = False
            time.sleep(0.05 if shift_pressed else 0.1)
        
        if glfw.get_key(window, glfw.KEY_HOME) == glfw.PRESS:
            current_frame = 0
            time.sleep(0.2)
        
        if glfw.get_key(window, glfw.KEY_END) == glfw.PRESS:
            current_frame = len(states) - 1
            time.sleep(0.2)
        
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            playback_speed = min(4.0, playback_speed * 1.2)
            print(f"   Speed: {playback_speed:.1f}x")
            time.sleep(0.1)
        
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            playback_speed = max(0.25, playback_speed / 1.2)
            print(f"   Speed: {playback_speed:.1f}x")
            time.sleep(0.1)
        
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            cam.azimuth = 180
            cam.elevation = -25
            cam.distance = 1.8
            cam.lookat[:] = [0.7, 0.0, 0.1]
            time.sleep(0.2)
        
        # Update frame
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        if playing:
            # Advance frame based on playback speed (30 fps base)
            frame_increment = max(1, int(30 * dt * playback_speed))
            current_frame += frame_increment
            if current_frame >= len(states):
                current_frame = 0  # Loop
        
        # Set state for rendering (NO SIMULATION)
        data.qpos[:] = states[current_frame]
        mujoco.mj_forward(model, data)
        
        # Render
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                              mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        
        # Draw HUD
        status = "▶ PLAYING" if playing else "⏸ PAUSED"
        speed_str = f"{playback_speed:.1f}x"
        progress_pct = (current_frame / (len(states) - 1)) * 100 if len(states) > 1 else 0
        
        hud_lines = [
            f"Frame: {current_frame}/{len(states)-1} ({progress_pct:.1f}%)",
            f"{status}  |  Speed: {speed_str}",
            f"Episode: {metadata.get('episode_id', 'N/A')}  |  Loss: {metadata.get('loss', 0):.4f}",
        ]
        
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            "\n".join(hud_lines).encode(),
            "".encode(),
            context
        )
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # Cleanup
    glfw.terminate()
    print("\n✅ Viewer closed")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Cloth replay viewer using pre-recorded states',
        epilog='First record states with: python replay_cloth_full.py <action_log.json> --save-states states.npz'
    )
    parser.add_argument('states_file', type=str, help='Path to pre-recorded states (.npz file)')
    args = parser.parse_args()
    
    states_path = Path(args.states_file)
    if not states_path.exists():
        print(f"❌ States file not found: {states_path}")
        print("\nFirst record states with:")
        print("  python replay_cloth_full.py <action_log.json> --save-states states.npz")
        sys.exit(1)
    
    # Load states
    print("=" * 70)
    print("  🎬 CLOTH REPLAY VIEWER (Pre-recorded States)")
    print("=" * 70)
    print(f"\n📂 Loading: {states_path}")
    
    states, metadata = load_states(states_path)
    
    print(f"\n📋 Episode: {metadata.get('episode_id', 'N/A')}")
    print(f"   Rope ID: {metadata.get('rope_id', 'N/A')}")
    print(f"   States: {len(states)}")
    print(f"   Final loss: {metadata.get('loss', 0):.4f}")
    
    # Create rendering model
    rope_config = metadata['rope_config']
    print(f"\n🔧 Creating rendering model...")
    print(f"   Cloth spacing: {rope_config['cloth_spacing']:.4f}")
    print(f"   Cloth density: {rope_config['cloth_density']:.2f}")
    
    model, data = create_rendering_model(rope_config)
    
    # Visualize
    visualize_states(model, data, states, metadata)


if __name__ == '__main__':
    main()
