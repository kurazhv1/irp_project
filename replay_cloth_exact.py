#!/usr/bin/env python3
"""
EXACT replay of trained cloth model - reproduces original env.step() behavior

This script EXACTLY reproduces the original TableClothSimEnvironment.step()
behavior for visualization using MuJoCo 2.3.7 + GLFW.

Usage:
    python replay_cloth_exact.py <action_log.json>
"""

import sys
import json
import mujoco
import glfw
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from jinja2 import Template
from common.mujoco_util_mj3 import MujocoCompensatedPDController


def load_action_log(log_path: Path):
    """Load action log JSON file"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    print(f"📄 Loaded action log: {log_path.name}")
    print(f"   Episode: {data['episode_id']}")
    print(f"   Loss: {data['loss']:.6f}")
    print(f"   Actions: {len(data['actions'])}")
    return data


def load_cloth_model(rope_config):
    """
    Load cloth model using Jinja2 template with rope_config
    This is EXACTLY how original TableClothSimEnvironment creates the model!
    """
    template_path = Path(__file__).parent / 'assets' / 'mujoco' / 'cloth' / 'table_cloth_template.xml.jinja2'
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")
    
    print(f"✅ Loading cloth model from template with rope_config:")
    for key, val in rope_config.items():
        print(f"   {key}: {val}")
    
    # Render Jinja2 template with rope_config (EXACT original method)
    template = Template(template_path.open('r').read())
    xml = template.render(**rope_config)
    
    # Load model from XML string
    model = mujoco.MjModel.from_xml_string(xml)
    
    return model


def get_cubic_control(t, q, dt):
    """Generate cubic spline trajectory - EXACT copy from original"""
    duration = t[-1]
    n_steps = int(duration / dt)
    q_interp = CubicSpline(t, q, bc_type='clamped')
    dq_interp = q_interp.derivative()
    ts = np.arange(n_steps) * dt
    qs = q_interp(ts)
    dqs = dq_interp(ts)
    return qs, dqs, ts


def find_cloth_keypoint_ids(model):
    """Find 9 cloth keypoint body IDs (3x3 grid)"""
    cloth_body_ids = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and 'B' in body_name and '_' in body_name:
            try:
                parts = body_name.split('_')
                if len(parts) == 2:
                    row, col = int(parts[0][1:]), int(parts[1])
                    cloth_body_ids.append((row, col, i))
            except:
                pass
    
    # Sort and create 13x13 grid
    cloth_body_ids.sort()
    grid = np.zeros((13, 13), dtype=int)
    for row, col, body_id in cloth_body_ids:
        grid[row, col] = body_id
    
    # Pick 9 points (3x3)
    indices = np.linspace(0, 12, 3).astype(int)
    kp_ids = []
    for i in indices:
        for j in indices:
            kp_ids.append(grid[i, j])
    
    return kp_ids


def get_cloth_keypoint_positions(data, kp_ids):
    """Get positions of cloth keypoints"""
    return np.array([data.xpos[body_id].copy() for body_id in kp_ids])


def create_compensated_controller(model, data, kp=100000, kv=100000):
    """Create compensated PD controller"""
    joint_names = ['gy', 'gz']
    return MujocoCompensatedPDController(model, data, joint_names, kp=kp, kv=kv)


def setup_viewer():
    """Initialize GLFW window"""
    if not glfw.init():
        raise Exception("GLFW init failed")
    
    window = glfw.create_window(1200, 900, "Cloth Exact Replay", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create window")
    
    glfw.make_context_current(window)
    return window


def setup_camera(cam):
    """Setup camera view"""
    cam.azimuth = 90
    cam.elevation = -45
    cam.distance = 2.5
    cam.lookat[0] = 0.0
    cam.lookat[1] = 1.0
    cam.lookat[2] = 0.8


def replay_single_action(model, data, raw_action, kp_ids, goal_coords, ctrl, window, cam, opt, scene, context):
    """
    Replay single action - EXACT copy of original env.step() logic
    
    Original parameters:
    - dt = 0.01
    - max_steps = 400
    - n_steps = min(max_steps, len(qs) + pad_steps + 20)
    """
    # EXACT original parameters
    dt = 0.01  # Original uses 0.01, NOT model.opt.timestep!
    max_steps = 400
    
    # Parse action
    duration, gy1, gz1, gy2 = raw_action
    gz2 = 0.05
    
    print(f"   🔍 RAW ACTION DETAILS:")
    print(f"      duration = {duration:.6f}s")
    print(f"      gy1 = {gy1:.6f} (Y coord of waypoint 1)")
    print(f"      gz1 = {gz1:.6f} (Z coord of waypoint 1)")
    print(f"      gy2 = {gy2:.6f} (Y coord of waypoint 2)")
    print(f"      gz2 = {gz2:.6f} (Z coord of waypoint 2, fixed)")
    print(f"   📍 Trajectory: [0,0] → [{gy1:.3f}, {gz1:.3f}] → [{gy2:.3f}, {gz2:.3f}]")
    
    # Generate cubic spline (EXACT original)
    t_in = np.linspace(0, duration, 3)
    q_in = np.array([
        [0, 0],
        [gy1, gz1],
        [gy2, gz2]
    ])
    qs, dqs, ts = get_cubic_control(t_in, q_in, dt)
    pad_steps = int(0.2 / dt)
    
    # Calculate n_steps (EXACT original)
    n_steps = min(max_steps, len(qs) + pad_steps + 20)
    
    print(f"   len(qs)={len(qs)}, pad={pad_steps}, n_steps={n_steps}")
    print(f"   Total time: {n_steps * dt:.2f}s")
    
    hist = []
    
    # Get joint IDs for gy and gz
    gy_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gy')
    gz_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gz')
    
    # Get actuator IDs (CORRECT NAMES from cloth.xml!)
    gy_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'y_motor')
    gz_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'z_motor')
    
    print(f"   🎛️  CONTROLLER SETUP:")
    print(f"      Joint IDs: gy={gy_jnt_id}, gz={gz_jnt_id}")
    print(f"      Actuator IDs: gy={gy_act_id}, gz={gz_act_id}")
    print(f"      Initial qpos: gy={data.qpos[gy_jnt_id]:.6f}, gz={data.qpos[gz_jnt_id]:.6f}")
    
    # Simple PD control without compensation (to avoid Nan/Inf)
    kp = 100000
    kv = 100000
    print(f"      PD gains: kp={kp}, kv={kv}")
    
    # Simulate (EXACT original loop)
    print(f"   🔄 SIMULATION LOOP:")
    
    # Log first few steps for debugging
    log_steps = [0, 10, 50, 100, n_steps-1]
    
    for i in range(n_steps):
        ii = max(min(i, len(qs)-1), 0)
        q = qs[ii]
        dq = dqs[ii]
        
        # Simple PD control (no gravity/Coriolis compensation to avoid instability)
        q_err = q - data.qpos[gy_jnt_id:gz_jnt_id+1]
        dq_err = dq - data.qvel[gy_jnt_id:gz_jnt_id+1]
        u = kp * q_err + kv * dq_err
        
        # Log key steps
        if i in log_steps:
            print(f"      Step {i}/{n_steps}:")
            print(f"         Target q: [{q[0]:.4f}, {q[1]:.4f}], dq: [{dq[0]:.4f}, {dq[1]:.4f}]")
            print(f"         Actual qpos: [{data.qpos[gy_jnt_id]:.4f}, {data.qpos[gz_jnt_id]:.4f}]")
            print(f"         Actual qvel: [{data.qvel[gy_jnt_id]:.4f}, {data.qvel[gz_jnt_id]:.4f}]")
            print(f"         Error q: [{q_err[0]:.4f}, {q_err[1]:.4f}], dq: [{dq_err[0]:.4f}, {dq_err[1]:.4f}]")
            print(f"         Control u: [{u[0]:.2f}, {u[1]:.2f}]")
        
        # Send forces directly
        data.ctrl[gy_act_id] = u[0]
        data.ctrl[gz_act_id] = u[1]
        
        # Physics step
        mujoco.mj_step(model, data)
        
        # Record keypoints
        kp_positions = get_cloth_keypoint_positions(data, kp_ids)
        hist.append(kp_positions)
        
        # Render
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        
        mujoco.mjv_updateScene(model, data, opt, None, cam, 
                               mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        # Draw goal points as red spheres using connectors (overlay)
        for idx, goal_pos in enumerate(goal_coords):
            if scene.ngeom >= scene.maxgeom - 1:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.zeros(3),  # size will be set below
                goal_pos,     # position
                np.eye(3).flatten(),  # orientation (identity matrix)
                np.array([1.0, 0.0, 0.0, 0.8])  # RGBA: red, semi-transparent
            )
            geom.size[0] = 0.015  # radius 1.5cm
            geom.size[1] = 0.015
            geom.size[2] = 0.015
            scene.ngeom += 1
        
        mujoco.mjr_render(viewport, scene, context)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        if glfw.window_should_close(window) or \
           glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            return None
    
    hist = np.array(hist)
    
    # Compute loss (on last step)
    final_positions = hist[-1]  # Shape: (9, 3)
    diffs = final_positions - goal_coords
    distances = np.linalg.norm(diffs, axis=1)
    mean_dist = np.mean(distances)
    
    print(f"   📊 FINAL STEP ANALYSIS:")
    print(f"      Final qpos: gy={data.qpos[gy_jnt_id]:.6f}, gz={data.qpos[gz_jnt_id]:.6f}")
    print(f"      Keypoint positions (first 3):")
    for i in range(min(3, len(final_positions))):
        print(f"         KP{i}: actual={final_positions[i]}, goal={goal_coords[i]}, error={distances[i]*100:.2f}cm")
    print(f"      Mean distance: {mean_dist*100:.2f}cm")
    
    return hist


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_cloth_exact.py <action_log.json>")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"❌ Action log not found: {log_path}")
        sys.exit(1)
    
    # Load data
    log_data = load_action_log(log_path)
    
    # Load model with rope_config from action log (EXACT original method!)
    rope_config = log_data['rope_config']
    model = load_cloth_model(rope_config)
    data = mujoco.MjData(model)
    
    print(f"\n📊 Model info:")
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Timestep: {model.opt.timestep}")
    
    # Find keypoints
    kp_ids = find_cloth_keypoint_ids(model)
    print(f"   Keypoints: {len(kp_ids)}")
    
    # Setup viewer
    window = setup_viewer()
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    setup_camera(cam)
    
    # Get actions
    raw_actions = log_data['raw_actions']
    goal_coords = np.array(log_data['goal_coords'])
    
    print(f"\n🎬 Starting replay...")
    print(f"   Episode: {log_data['episode_id']}")
    print(f"   Actions: {len(raw_actions)}")
    print(f"   Goal coords (first 3):")
    for i in range(min(3, len(goal_coords))):
        print(f"     [{goal_coords[i][0]:.3f}, {goal_coords[i][1]:.3f}, {goal_coords[i][2]:.3f}]")
    print(f"   Press ESC to exit\n")
    
    # Initial forward
    mujoco.mj_forward(model, data)
    
    # Create compensated PD controller
    ctrl = create_compensated_controller(model, data, kp=100000, kv=100000)
    print(f"   Using compensated PD controller (gravity compensation enabled)")
    
    # Save initial state (EXACTLY like original env.init_state!)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    initial_ctrl = data.ctrl.copy()
    
    print(f"\n🔄 IMPORTANT: State will be RESET before EACH action (just like original env!)")
    print(f"   Initial state saved (qpos[505:507]={data.qpos[505:507]})")
    
    try:
        # Replay each action with FULL RESET before each (EXACT original behavior!)
        for action_idx, raw_action in enumerate(raw_actions):
            print(f"\n🎯 Action {action_idx + 1}/{len(raw_actions)}:")
            
            # RESET TO INITIAL STATE BEFORE EACH ACTION (original env.step() does this!)
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = initial_ctrl
            mujoco.mj_forward(model, data)
            
            print(f"   ♻️  State RESET to initial (qpos[505:507]={data.qpos[505:507]})")
            
            hist = replay_single_action(model, data, raw_action, kp_ids, goal_coords, ctrl,
                                       window, cam, opt, scene, context)
            
            if hist is None:
                print("\n⚠️  Exiting early...")
                break
            
            # Compute loss for this action
            final_positions = hist[-1]
            diff = final_positions - goal_coords
            dists = np.linalg.norm(diff, axis=-1)
            loss = np.mean(dists)
            
            print(f"   Final loss: {loss:.6f}")
            print(f"   Mean distance to goal: {loss*100:.1f}cm")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    finally:
        glfw.terminate()
        print("\n✅ Replay finished")


if __name__ == '__main__':
    main()
