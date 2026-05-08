#!/usr/bin/env python3
"""
Simple viewer for pre-recorded cloth states.
No simulation - just playback of states saved by replay_cloth_full.py

This ensures CORRECT physics visualization (mujoco-py) without segfaults.
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
    print(f"📂 Loading: {states_path}")
    data = np.load(states_path, allow_pickle=True)
    states = data['states']
    metadata = data['metadata'].item()
    print(f"   ✓ Loaded {len(states)} states")
    return states, metadata


def create_rendering_model(rope_config):
    """Create MuJoCo model for rendering only (no simulation)"""
    template_path = Path('assets/mujoco/cloth/table_cloth_template.xml.jinja2')
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    # Render template with rope_config (contains all needed parameters)
    xml_string = template.render(**rope_config)  # Unpack dictionary
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
    window = glfw.create_window(width, height, "🎬 Cloth Replay - Correct Physics", None, None)
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
    current_frame = 0.0  # Float for smooth interpolation
    playing = False  # Start paused so user can see initial state
    playback_speed = 2.0  # Start at 2x speed (since base is slow)
    last_time = time.time()
    
    print(f"   Total frames: {len(states)}")
    print(f"   Actions: {metadata.get('n_actions', 'N/A')}")
    print("\n💡 Controls:")
    print("   Space:           Pause/Play")
    print("   Left/Right:      Previous/Next action")
    print("   Shift+Left/Right: -10/+10 frames")
    print("   Up/Down:         Speed up/down")
    print("   Home:            Jump to start")
    print("   End:             Jump to end")
    print("   R:               Reset camera")
    print("   ESC:             Exit")
    print("=" * 70)
    print("\n▶️  Press SPACE to start playback")
    print("=" * 70)
    
    # Main render loop
    frame_hold_time = 0
    last_key_time = 0
    
    while not glfw.window_should_close(window):
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Handle keyboard input with debouncing
        key_cooldown = 0.15
        can_press_key = (current_time - last_key_time) > key_cooldown
        
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and can_press_key:
            playing = not playing
            last_key_time = current_time
        
        shift_pressed = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS and can_press_key:
            step = 10.0 if shift_pressed else 1.0
            current_frame = max(0.0, current_frame - step)
            playing = False
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS and can_press_key:
            step = 10.0 if shift_pressed else 1.0
            current_frame = min(len(states) - 1.0, current_frame + step)
            playing = False
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_HOME) == glfw.PRESS and can_press_key:
            current_frame = 0.0
            playing = False
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_END) == glfw.PRESS and can_press_key:
            current_frame = float(len(states) - 1)
            playing = False
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS and can_press_key:
            playback_speed = min(4.0, playback_speed * 1.5)
            print(f"   ⏩ Speed: {playback_speed:.1f}x")
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS and can_press_key:
            playback_speed = max(0.25, playback_speed / 1.5)
            print(f"   ⏪ Speed: {playback_speed:.1f}x")
            last_key_time = current_time
        
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS and can_press_key:
            cam.azimuth = 180
            cam.elevation = -25
            cam.distance = 1.8
            cam.lookat[:] = [0.7, 0.0, 0.1]
            print("   🔄 Camera reset")
            last_key_time = current_time
        
        # Update frame with INTERPOLATION for smooth playback
        if playing:
            # Advance slowly - we only have 15 keyframes for 16 actions
            # Each action should take ~2 seconds to show smooth interpolation
            # So: 15 frames over ~30 seconds = 0.5 fps
            frame_hold_time += dt
            frames_to_advance = frame_hold_time * 0.5 * playback_speed  # 0.5 fps base (2 sec per frame)
            if frames_to_advance > 0.01:  # Smooth fractional advancement
                current_frame += frames_to_advance
                frame_hold_time = 0
                
                if current_frame >= len(states) - 1:
                    current_frame = 0  # Loop back to start
        
        # Clamp frame (allow fractional for interpolation)
        current_frame = max(0.0, min(len(states) - 1.0, current_frame))
        
        # INTERPOLATE between saved states for smooth animation
        frame_idx = int(current_frame)
        frame_frac = current_frame - frame_idx
        
        if frame_idx >= len(states) - 1:
            # Last frame - no interpolation
            data.qpos[:] = states[-1]
        else:
            # Linear interpolation between frame_idx and frame_idx+1
            state_a = states[frame_idx]
            state_b = states[frame_idx + 1]
            interpolated_state = state_a * (1.0 - frame_frac) + state_b * frame_frac
            data.qpos[:] = interpolated_state
        
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
        
        action_num = int(current_frame) + 1  # Approximate action number
        
        hud_lines = [
            f"Action: {action_num}/{metadata.get('n_actions', '?')}  |  Frame: {current_frame:.1f}/{len(states)-1} ({progress_pct:.1f}%)",
            f"{status}  |  Speed: {speed_str}  |  [INTERPOLATED]",
            f"Episode: {metadata.get('episode_id', 'N/A')}  |  Rope {metadata.get('rope_id', '?')}  |  Final Loss: {metadata.get('loss', 0)*100:.2f}cm",
        ]
        
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            "\n".join(hud_lines).encode(),
            "Physics: mujoco-py (correct)".encode(),
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
        print("\n💡 First record states with:")
        print("  python replay_cloth_full.py <action_log.json> --save-states states.npz")
        sys.exit(1)
    
    # Load states
    print("=" * 70)
    print("  🎬 CLOTH REPLAY VIEWER")
    print("  ✅ Using pre-recorded states (correct physics)")
    print("=" * 70)
    
    states, metadata = load_states(states_path)
    
    print(f"\n📋 Episode: {metadata.get('episode_id', 'N/A')}")
    print(f"   Rope ID: {metadata.get('rope_id', 'N/A')}")
    print(f"   Actions: {metadata.get('n_actions', 'N/A')}")
    print(f"   States: {len(states)}")
    print(f"   Final loss: {metadata.get('loss', 0):.4f} ({metadata.get('loss', 0)*100:.2f}cm)")
    
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
