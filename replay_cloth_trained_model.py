#!/usr/bin/env python3
"""
Replay trained cloth model predictions with GLFW visualization

Uses working MuJoCo 2.3.7 + GLFW approach for stable visualization.
Loads action logs from eval_irp_cloth_sim.py and replays them with cloth physics.

Usage:
    python replay_cloth_trained_model.py <action_log.json>
    
Example:
    python replay_cloth_trained_model.py output/20241130_123456/action_log_rope0_goal5.json
"""

import sys
import json
import mujoco
import glfw
import numpy as np
from pathlib import Path
from jinja2 import Template
from scipy.interpolate import CubicSpline


def load_action_log(log_path: Path):
    """Load action log JSON file"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    print(f"📄 Loaded action log: {log_path.name}")
    print(f"   Episode: {data['episode_id']}")
    print(f"   Rope params: spacing={data['rope_config']['cloth_spacing']:.3f}, "
          f"density={data['rope_config']['cloth_density']:.3f}")
    print(f"   Goal alpha: {data['goal_alpha']:.3f}")
    print(f"   Loss: {data['loss']:.6f}")
    print(f"   Actions: {len(data['actions'])}")
    return data


def create_model_from_config(rope_config):
    """Create MuJoCo model - uses static cloth.xml for correct visuals"""
    # Always use static cloth.xml for correct skybox and floor appearance
    xml_path = Path(__file__).parent / 'assets' / 'mujoco' / 'cloth' / 'cloth.xml'
    
    if not xml_path.exists():
        print(f"❌ cloth.xml not found at {xml_path}")
        raise FileNotFoundError(f"cloth.xml not found at {xml_path}")
    
    print(f"✅ Loading static cloth.xml (original visuals)")
    print(f"   Note: Using fixed cloth parameters (spacing=0.05, density from XML)")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    
    return model


def setup_viewer():
    """Initialize GLFW window and rendering context"""
    if not glfw.init():
        raise Exception("❌ GLFW init failed")
    
    window = glfw.create_window(1200, 900, "Cloth Trained Model Replay", None, None)
    if not window:
        glfw.terminate()
        raise Exception("❌ Failed to create GLFW window")
    
    glfw.make_context_current(window)
    print("✅ GLFW window created (1200×900)")
    
    return window


def setup_camera(cam):
    """Setup camera for good cloth view"""
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 2.5
    cam.lookat[:] = [0.5, 0.8, 0.3]


def get_cubic_control(t, q, dt):
    """
    Generate cubic spline trajectory for PD control
    
    Args:
        t: input time steps (T)
        q: input joint pos (T,Q)
        dt: timestep
    
    Returns:
        qs: output joint pos steps (N,Q)
        dqs: output joint vel steps (N,Q)
        ts: output time steps (N)
    """
    duration = t[-1]
    n_steps = int(duration / dt)
    q_interp = CubicSpline(t, q, bc_type='clamped')
    dq_interp = q_interp.derivative()
    ts = np.arange(n_steps) * dt
    qs = q_interp(ts)
    dqs = dq_interp(ts)
    return qs, dqs, ts


def simple_pd_control(model, data, q_target, dq_target, kp=100000, kv=100000):
    """
    Simple PD controller for gripper joints
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        q_target: target positions [gy, gz]
        dq_target: target velocities [dgy, dgz]
        kp: position gain
        kv: velocity gain
    """
    # Get joint IDs for gy and gz
    gy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gy')
    gz_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gz')
    
    # Current positions and velocities
    q_current = np.array([data.qpos[gy_id], data.qpos[gz_id]])
    dq_current = np.array([data.qvel[gy_id], data.qvel[gz_id]])
    
    # PD control
    u = kp * (q_target - q_current) + kv * (dq_target - dq_current)
    
    # Apply control
    data.ctrl[0] = u[0]  # y_motor
    data.ctrl[1] = u[1]  # z_motor


def get_cloth_keypoint_positions(data, kp_ids):
    """Get positions of cloth keypoints"""
    return np.array([data.xpos[body_id].copy() for body_id in kp_ids])


def find_cloth_keypoint_ids(model):
    """Find cloth body IDs for 9 keypoints (3x3 grid)"""
    # Find all cloth bodies
    cloth_body_ids = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and 'B' in body_name and '_' in body_name:
            cloth_body_ids.append(i)
    
    if len(cloth_body_ids) == 0:
        print("⚠️  No cloth bodies found, using default IDs")
        return list(range(9))
    
    # Try to reconstruct 13x13 grid and pick 3x3 subset
    cloth_body_ids = np.array(cloth_body_ids)
    
    if len(cloth_body_ids) >= 169:  # 13x13 grid
        cloth_grid = cloth_body_ids[:169].reshape(13, 13)
        # Pick 9 points: corners, edges, center
        coords = [(0, 0), (0, 6), (0, 12),
                  (6, 0), (6, 6), (6, 12),
                  (12, 0), (12, 6), (12, 12)]
        kp_ids = [cloth_grid[i, j] for i, j in coords]
        print(f"✅ Found 9 keypoints from 13×13 cloth grid")
    else:
        # Fallback: use first 9 cloth bodies
        kp_ids = cloth_body_ids[:9].tolist()
        print(f"⚠️  Using first 9 cloth bodies as keypoints")
    
    return kp_ids


def replay_with_visualization(log_data: dict):
    """Main replay function with GLFW visualization"""
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model_from_config(log_data['rope_config'])
    data = mujoco.MjData(model)
    
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Actuators: {model.nu}")
    
    # Find cloth keypoints
    kp_ids = find_cloth_keypoint_ids(model)
    
    # Setup viewer
    print("\n🖼️  Setting up viewer...")
    window = setup_viewer()
    
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    setup_camera(cam)
    print("✅ Rendering context ready")
    
    # Prepare action trajectory
    actions = log_data['actions']
    raw_actions = log_data['raw_actions']
    
    print(f"\n🎬 Starting replay...")
    print(f"   Episode: {log_data['episode_id']}")
    print(f"   Actions to replay: {len(actions)}")
    print(f"   Press ESC to exit early\n")
    
    # Simulation parameters (use model's timestep from XML)
    dt = model.opt.timestep  # Use actual timestep from cloth.xml (0.002)
    kp_gain = 100000
    kv_gain = 100000
    
    print(f"   Using dt={dt} from model")
    print(f"   PD gains: kp={kp_gain}, kv={kv_gain}")
    
    trajectory_history = []
    
    try:
        # Initial forward
        mujoco.mj_forward(model, data)
        
        # Process each action sequentially (NO RESET - cumulative effect)
        for action_idx, raw_action in enumerate(raw_actions):
            duration, gy1, gz1, gy2 = raw_action
            gz2 = 0.05  # Fixed final height
            
            print(f"\n🎯 Action {action_idx + 1}/{len(raw_actions)}:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Trajectory: [0,0] → [{gy1:.3f}, {gz1:.3f}] → [{gy2:.3f}, {gz2:.3f}]")
            
            # Generate cubic spline trajectory
            t_in = np.linspace(0, duration, 3)
            q_in = np.array([
                [0, 0],        # Start position
                [gy1, gz1],    # Mid position
                [gy2, gz2]     # End position
            ])
            qs, dqs, ts = get_cubic_control(t_in, q_in, dt)
            pad_steps = int(0.2 / dt)  # 200ms settling time
            max_steps_per_action = 2000  # Allow more steps than original
            
            n_steps = min(max_steps_per_action, len(qs) + pad_steps + 20)
            
            print(f"   Trajectory steps: {len(qs)}, padding: {pad_steps}, total: {n_steps}")
            print(f"   Total time: {n_steps * dt:.2f}s")
            
            # Execute trajectory with PD control
            for i in range(n_steps):
                # Get target from trajectory (hold last position during padding)
                ii = min(i, len(qs) - 1)
                q_target = qs[ii]
                dq_target = dqs[ii]
                
                # Apply PD control
                simple_pd_control(model, data, q_target, dq_target, kp_gain, kv_gain)
                
                # Physics step
                mujoco.mj_step(model, data)
                
                # Record trajectory
                kp_positions = get_cloth_keypoint_positions(data, kp_ids)
                trajectory_history.append(kp_positions)
                
                # Render every frame
                w, h = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, w, h)
                
                mujoco.mjv_updateScene(model, data, opt, None, cam, 
                                       mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, context)
                
                glfw.swap_buffers(window)
                glfw.poll_events()
                
                # Check for early exit
                if glfw.window_should_close(window) or \
                   glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    print("\n⚠️  Exiting early...")
                    raise KeyboardInterrupt
        
        print(f"\n✅ Replay completed!")
        print(f"   Total steps: {len(trajectory_history)}")
        print(f"   Actions replayed: {len(raw_actions)}/{len(raw_actions)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        
    # Compute final loss (distance from goal)
    try:
        if len(trajectory_history) > 0:
            final_positions = trajectory_history[-1]
            goal_coords = np.array(log_data.get('goal_coords', []))
            if len(goal_coords) > 0:
                dists = np.linalg.norm(final_positions - goal_coords, axis=-1)
                mean_dist = np.mean(dists)
                print(f"   Final loss (mean dist): {mean_dist:.6f}m")
                print(f"   Original logged loss: {log_data['loss']:.6f}")
    except Exception as e:
        pass
        
    except Exception as e:
        print(f"\n❌ Error during replay: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Cleaning up...")
        glfw.terminate()


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_cloth_trained_model.py <action_log.json>")
        print("\nExample:")
        print("  python replay_cloth_trained_model.py output/20241130_123456/action_log_rope0_goal5.json")
        print("\nAvailable logs:")
        output_dir = Path(__file__).parent / 'output'
        if output_dir.exists():
            for log_file in sorted(output_dir.glob('*/action_log_*.json'))[:5]:
                print(f"  {log_file}")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    
    if not log_path.exists():
        print(f"❌ File not found: {log_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("  CLOTH TRAINED MODEL REPLAY WITH VISUALIZATION")
    print("=" * 70)
    print()
    
    # Load action log
    log_data = load_action_log(log_path)
    
    # Replay with visualization
    replay_with_visualization(log_data)
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
