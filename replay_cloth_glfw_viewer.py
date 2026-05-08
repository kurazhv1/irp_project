#!/usr/bin/env python3
"""
Cloth Replay Viewer - Interactive visualization using MuJoCo 2.3.7 + GLFW

Replays cloth manipulation with interactive viewer:
- Timeline slider to navigate through replay
- Real cloth physics visualization  
- Goal points overlay
- Pause/play controls

Usage:
    python replay_cloth_glfw_viewer.py <action_log.json>
"""

import sys
import json
import mujoco
import glfw
import numpy as np
from pathlib import Path
from jinja2 import Template


def load_action_log(log_path):
    """Load action log JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def create_cloth_model(rope_config, stiffer=True):
    """
    Create cloth model from template with rope_config
    
    Args:
        rope_config: Cloth configuration parameters
        stiffer: If True, make cloth stiffer (less floppy)
    """
    template_path = Path(__file__).parent / 'assets/mujoco/cloth/table_cloth_template.xml.jinja2'
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    xml_content = template_path.open('r').read()
    
    if stiffer:
        # Make cloth MUCH stiffer - less deformation, more rigid behavior
        print(f"   🔧 Making cloth stiffer...")
        # Increase flatinertia (0.01 -> 0.1): 10x more resistant to bending
        xml_content = xml_content.replace('flatinertia="0.01"', 'flatinertia="0.1"')
        # Increase joint damping (0.001 -> 0.01): 10x more damping
        xml_content = xml_content.replace('damping="0.001"', 'damping="0.01"')
        # Increase twist damping (0.0001 -> 0.001): 10x more twist resistance
        xml_content = xml_content.replace('damping="0.0001"', 'damping="0.001"')
    
    template = Template(xml_content)
    xml = template.render(**rope_config)
    model = mujoco.MjModel.from_xml_string(xml)
    
    return model


def simulate_action(model, data, raw_action, dt=0.01, max_steps=400):
    """
    Simulate single action and record states
    
    Returns list of qpos states for each timestep
    """
    duration, gy1, gz1, gy2 = raw_action
    gz2 = 0.05
    
    # Generate cubic spline trajectory
    from scipy.interpolate import CubicSpline
    
    t_in = np.linspace(0, duration, 3)
    q_in = np.array([[0, 0], [gy1, gz1], [gy2, gz2]])
    
    q_interp = CubicSpline(t_in, q_in, bc_type='clamped')
    dq_interp = q_interp.derivative()
    
    n_steps = int(duration / dt)
    ts = np.arange(n_steps) * dt
    qs = q_interp(ts)
    dqs = dq_interp(ts)
    
    pad_steps = int(0.2 / dt)
    n_steps = min(max_steps, len(qs) + pad_steps + 20)
    
    # Get joint and actuator IDs
    gy_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gy')
    gz_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gz')
    gy_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'y_motor')
    gz_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'z_motor')
    
    kp = 100000
    kv = 100000
    
    states = []
    
    for i in range(n_steps):
        ii = max(min(i, len(qs)-1), 0)
        q = qs[ii]
        dq = dqs[ii]
        
        # PD control
        q_err = q - data.qpos[gy_jnt:gz_jnt+1]
        dq_err = dq - data.qvel[gy_jnt:gz_jnt+1]
        u = kp * q_err + kv * dq_err
        
        data.ctrl[gy_act] = u[0]
        data.ctrl[gz_act] = u[1]
        
        mujoco.mj_step(model, data)
        
        # Save state
        states.append(data.qpos.copy())
    
    return states


def precompute_replay_states(model, data, actions):
    """Precompute all states for entire replay"""
    print(f"\n⏳ Precomputing replay states...")
    
    all_states = []
    
    # Save initial state (only for return value)
    initial_qpos = data.qpos.copy()
    
    # NO RESET between actions - state accumulates sequentially!
    for i, raw_action in enumerate(actions):
        print(f"   Action {i+1}/{len(actions)}...", end='\r')
        
        # Simulate and record states (starting from previous end state)
        states = simulate_action(model, data, raw_action)
        all_states.extend(states)
    
    print(f"\n✅ Precomputed {len(all_states)} states")
    return all_states, initial_qpos


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive cloth replay viewer')
    parser.add_argument('action_log', type=str, help='Path to action log JSON file')
    parser.add_argument('--soft', action='store_true', help='Use soft cloth (original physics)')
    args = parser.parse_args()
    
    log_path = Path(args.action_log)
    if not log_path.exists():
        print(f"❌ Action log not found: {log_path}")
        sys.exit(1)
    
    stiffer_cloth = not args.soft  # Default: stiffer
    
    # Load action log
    action_log = load_action_log(log_path)
    metadata = action_log['metadata']
    raw_actions = action_log['raw_actions']
    goal_coords = np.array(action_log['goal_coords'])
    
    print("=" * 70)
    print("  🎬 CLOTH REPLAY VIEWER")
    print("=" * 70)
    print(f"\n📋 Episode: {action_log['episode_id']}")
    print(f"   Rope ID: {metadata['rope_id']}")
    print(f"   Actions: {len(raw_actions)}")
    print(f"   Final loss: {action_log['loss']:.4f} ({action_log['loss']*100:.2f}cm)")
    print(f"   Cloth mode: {'SOFT (original)' if args.soft else 'STIFF (rigid)'}")
    
    # Create model
    rope_param = metadata['rope_param']
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,
        'cloth_density': rope_param[1],
    }
    
    print(f"\n🔧 Creating cloth model...")
    model = create_cloth_model(rope_config, stiffer=stiffer_cloth)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Precompute all replay states
    all_states, initial_qpos = precompute_replay_states(model, data, raw_actions)
    
    # Initialize GLFW
    if not glfw.init():
        raise Exception("❌ GLFW init failed")
    
    window = glfw.create_window(1400, 900, "Cloth Replay Viewer", None, None)
    if not window:
        glfw.terminate()
        raise Exception("❌ Failed to create window")
    
    glfw.make_context_current(window)
    
    # Setup MuJoCo rendering
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 2.5
    cam.lookat[0] = 0.0
    cam.lookat[1] = 1.0
    cam.lookat[2] = 0.8
    
    print(f"\n🎥 Starting interactive viewer...")
    print(f"   Total frames: {len(all_states)}")
    print(f"\n💡 Controls:")
    print(f"   Space:      Pause/Play")
    print(f"   Left/Right: Step backward/forward")
    print(f"   Home:       Jump to start")
    print(f"   End:        Jump to end")
    print(f"   ESC:        Exit")
    print("=" * 70 + "\n")
    
    # Playback state
    current_frame = 0
    playing = False
    playback_speed = 1  # frames per render loop
    
    while not glfw.window_should_close(window):
        # Handle keyboard input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            playing = not playing
            glfw.wait_events_timeout(0.2)  # Debounce
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            current_frame = min(current_frame + 1, len(all_states) - 1)
            playing = False
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            current_frame = max(current_frame - 1, 0)
            playing = False
        if glfw.get_key(window, glfw.KEY_HOME) == glfw.PRESS:
            current_frame = 0
            playing = False
        if glfw.get_key(window, glfw.KEY_END) == glfw.PRESS:
            current_frame = len(all_states) - 1
            playing = False
        
        # Auto-advance if playing
        if playing:
            current_frame += playback_speed
            if current_frame >= len(all_states):
                current_frame = 0  # Loop
        
        current_frame = np.clip(current_frame, 0, len(all_states) - 1)
        
        # Set state
        data.qpos[:] = all_states[current_frame]
        mujoco.mj_forward(model, data)
        
        # Render
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        # Draw goal points
        for goal_pos in goal_coords:
            if scene.ngeom >= scene.maxgeom - 1:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.zeros(3),
                goal_pos,
                np.eye(3).flatten(),
                np.array([1.0, 0.0, 0.0, 0.8])  # Red
            )
            geom.size[0] = 0.015
            geom.size[1] = 0.015
            geom.size[2] = 0.015
            scene.ngeom += 1
        
        mujoco.mjr_render(viewport, scene, context)
        
        # Draw HUD text
        status = "PLAYING" if playing else "PAUSED"
        progress = f"{current_frame}/{len(all_states)}"
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, f"{status} | Frame: {progress}", 
                       context, 0.05, 0.95, 1, 1, 1)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()
    print("\n✅ Viewer closed")


if __name__ == '__main__':
    main()
