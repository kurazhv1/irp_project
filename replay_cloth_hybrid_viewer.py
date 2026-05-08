#!/usr/bin/env python3
"""
Hybrid cloth replay viewer:
- Simulation: mujoco-py (correct physics from TableClothSimEnvironment)
- Rendering: MuJoCo 2.3.7 via GLFW (stable, no segfault)

This approach ensures correct physics while having stable visualization.
"""

import json
import pathlib
import sys
import numpy as np
from pathlib import Path
import glfw
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Import mujoco-py for simulation
import mujoco_py as mj
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

# Import new MuJoCo for rendering only
import mujoco

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def load_action_log(log_path):
    """Load action log from JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def create_rendering_model(rope_config):
    """Create MuJoCo 2.3.7 model ONLY for rendering (not simulation)"""
    from jinja2 import Template
    
    template_path = Path('assets/mujoco/cloth/table_cloth_template.xml.jinja2')
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    xml_string = template.render(rope_config=rope_config)
    render_model = mujoco.MjModel.from_xml_string(xml_string)
    render_data = mujoco.MjData(render_model)
    
    return render_model, render_data


def simulate_with_real_physics(env, actions):
    """
    Simulate using REAL TableClothSimEnvironment (mujoco-py physics).
    Returns list of qpos states for visualization.
    """
    print("\n⏳ Simulating with REAL cloth physics (mujoco-py)...")
    
    all_states = []
    
    # Environment starts in initial state (no reset() method)
    all_states.append(env.sim.data.qpos.copy())
    
    n_actions = len(actions)
    for i, raw_action in enumerate(actions):
        print(f"   Action {i+1}/{n_actions}...", end='\r')
        
        # Execute action with real physics
        # raw_action is already [duration, gy1, gz1, gy2]
        obs, reward, done, info = env.step(raw_action)
        
        # Record final state after action completes
        all_states.append(env.sim.data.qpos.copy())
    
    print(f"\n✅ Simulated {len(all_states)} states with REAL physics")
    return all_states


def visualize_states(render_model, render_data, states, goal_coords):
    """
    Visualize pre-simulated states using MuJoCo 2.3.7 rendering.
    No simulation here - just playback of recorded states.
    """
    print("\n🎥 Starting interactive viewer (MuJoCo 2.3.7 rendering)...")
    
    # Initialize GLFW
    if not glfw.init():
        print("❌ Failed to initialize GLFW")
        sys.exit(1)
    
    # Create window
    width, height = 1280, 720
    window = glfw.create_window(width, height, "🎬 Cloth Replay (Hybrid Physics)", None, None)
    if not window:
        glfw.terminate()
        print("❌ Failed to create GLFW window")
        sys.exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Create MuJoCo rendering context
    cam = mujoco.MjvCamera()
    cam.azimuth = 180
    cam.elevation = -20
    cam.distance = 1.5
    cam.lookat[:] = [0.6, 0.0, 0.0]
    
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(render_model, maxgeom=10000)
    context = mujoco.MjrContext(render_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Add goal points as visual markers
    for goal in goal_coords:
        geom_id = mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_GEOM, 'goal_marker')
        if geom_id >= 0:
            render_model.geom_rgba[geom_id] = [1, 0, 0, 0.5]
    
    # Playback state
    current_frame = 0
    playing = True
    playback_speed = 1.0
    last_time = time.time()
    
    print(f"   Total frames: {len(states)}")
    print("\n💡 Controls:")
    print("   Space:      Pause/Play")
    print("   Left/Right: Step backward/forward")
    print("   Up/Down:    Speed up/down")
    print("   Home:       Jump to start")
    print("   End:        Jump to end")
    print("   ESC:        Exit")
    print("=" * 70)
    
    # Main render loop
    while not glfw.window_should_close(window):
        # Handle input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            playing = not playing
            time.sleep(0.2)  # Debounce
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            current_frame = max(0, current_frame - 1)
            playing = False
            time.sleep(0.05)
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            current_frame = min(len(states) - 1, current_frame + 1)
            playing = False
            time.sleep(0.05)
        if glfw.get_key(window, glfw.KEY_HOME) == glfw.PRESS:
            current_frame = 0
            time.sleep(0.2)
        if glfw.get_key(window, glfw.KEY_END) == glfw.PRESS:
            current_frame = len(states) - 1
            time.sleep(0.2)
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            playback_speed = min(4.0, playback_speed * 1.2)
            time.sleep(0.1)
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            playback_speed = max(0.25, playback_speed / 1.2)
            time.sleep(0.1)
        
        # Update frame
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        if playing:
            # Advance frame based on playback speed
            frame_increment = int(30 * dt * playback_speed)  # 30 fps base
            current_frame += frame_increment
            if current_frame >= len(states):
                current_frame = len(states) - 1
                playing = False
        
        # Set state for rendering (NO SIMULATION, just display)
        render_data.qpos[:] = states[current_frame]
        mujoco.mj_forward(render_model, render_data)
        
        # Render
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        mujoco.mjv_updateScene(render_model, render_data, opt, None, cam,
                              mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)
        
        # Draw HUD
        status = "▶ PLAYING" if playing else "⏸ PAUSED"
        speed_str = f"Speed: {playback_speed:.1f}x"
        hud_text = f"Frame: {current_frame}/{len(states)-1}  |  {status}  |  {speed_str}"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            viewport,
            hud_text.encode(),
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
    parser = argparse.ArgumentParser(description='Hybrid cloth replay viewer (mujoco-py sim + MuJoCo 2.3.7 rendering)')
    parser.add_argument('action_log', type=str, help='Path to action log JSON file')
    args = parser.parse_args()
    
    log_path = Path(args.action_log)
    if not log_path.exists():
        print(f"❌ Action log not found: {log_path}")
        sys.exit(1)
    
    # Load action log
    action_log = load_action_log(log_path)
    metadata = action_log['metadata']
    raw_actions = action_log['raw_actions']
    goal_coords = np.array(action_log['goal_coords'])
    
    print("=" * 70)
    print("  🎬 HYBRID CLOTH REPLAY VIEWER")
    print("=" * 70)
    print(f"\n📋 Episode: {action_log['episode_id']}")
    print(f"   Rope ID: {metadata['rope_id']}")
    print(f"   Actions: {len(raw_actions)}")
    print(f"   Final loss: {action_log['loss']:.4f} ({action_log['loss']*100:.2f}cm)")
    print(f"\n🔧 Physics: mujoco-py 2.1.2.14 (REAL TableClothSimEnvironment)")
    print(f"   Rendering: MuJoCo 2.3.7 (stable GLFW)")
    
    # Create REAL simulation environment (mujoco-py)
    rope_param = metadata['rope_param']
    goal_alpha = metadata['goal_alpha']
    
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
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        obs_topdown=False,
        show_vis=False  # Headless simulation
    )
    
    # Set up goal
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    
    # Simulate with REAL physics (mujoco-py)
    states = simulate_with_real_physics(env, raw_actions)
    
    # Create rendering model (MuJoCo 2.3.7) - same config as simulation
    render_model, render_data = create_rendering_model(rope_config)
    
    # Visualize using recorded states
    visualize_states(render_model, render_data, states, goal_coords)


if __name__ == '__main__':
    main()
