#!/usr/bin/env python
"""
Render cloth trajectories using MuJoCo 2.3.7 with GLFW
Loads pre-computed trajectories from mujoco-py simulation
and renders them with proper cloth visualization
"""

import sys
import pickle
import pathlib
import numpy as np
import mujoco
import glfw
from scipy.spatial.transform import Rotation
import imageio


def init_glfw_window(width=1920, height=1080):
    """Initialize GLFW window for rendering"""
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)  # 4x MSAA
    
    window = glfw.create_window(width, height, "Cloth Manipulation - IRP Project", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # VSync
    
    return window


def render_cloth_trajectory(data_path: str, output_video: str = None, show_window: bool = True):
    """
    Render cloth manipulation using MuJoCo 2.3.7
    
    Args:
        data_path: Path to .pkl file with trajectory data
        output_video: Output video path
        show_window: Whether to show live window
    """
    
    # Load trajectory data
    print(f"📂 Loading trajectory data: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    episode_id = data['episode_id']
    trajectories = data['trajectories']
    goal_coords = data['goal_coords']
    rope_config = data['rope_config']
    logged_loss = data['logged_loss']
    replayed_loss = data['replayed_final_loss']
    
    print(f"\n📊 Episode: {episode_id}")
    print(f"   Actions: {len(trajectories)}")
    print(f"   Logged loss: {logged_loss:.6f} ({logged_loss * 100:.2f}cm)")
    print(f"   Replayed loss: {replayed_loss:.6f} ({replayed_loss * 100:.2f}cm)")
    
    # Load cloth model
    print(f"\n🏗️  Loading cloth model...")
    cloth_spacing = rope_config['cloth_spacing']
    cloth_density = rope_config['cloth_density']
    
    # Generate cloth XML
    from jinja2 import Template
    xml_template_path = pathlib.Path(__file__).parent / 'assets/mujoco/cloth/table_cloth_template.xml.jinja2'
    
    with open(xml_template_path, 'r') as f:
        template = Template(f.read())
    
    xml_str = template.render(
        table_height=0.8,
        table_y=1.0,
        table_size=1.2,
        cloth_spacing=cloth_spacing,
        cloth_density=cloth_density
    )
    
    # Save temporary XML
    temp_xml = pathlib.Path('temp_cloth_render.xml')
    temp_xml.write_text(xml_str)
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(temp_xml))
    data_mj = mujoco.MjData(model)
    
    print(f"✅ Model loaded:")
    print(f"   Bodies: {model.nbody}")
    print(f"   Timestep: {model.opt.timestep}")
    
    # Get cloth body IDs (same keypoint selection as original)
    cloth_body_ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and 'B' in name and 'G' in name:  # Cloth bodies
            try:
                row = int(name.split('B')[1].split('_')[0])
                col = int(name.split('G')[1])
                cloth_body_ids.append((i, row, col))
            except:
                pass
    
    # Select 9 keypoints (3x3 grid)
    cloth_array = np.zeros((13, 13), dtype=int) - 1
    for body_id, row, col in cloth_body_ids:
        cloth_array[row, col] = body_id
    
    # Pick 9 points (corners, edges, center)
    keypoint_indices = []
    for i in [0, 6, 12]:
        for j in [0, 6, 12]:
            keypoint_indices.append(cloth_array[i, j])
    
    print(f"   Keypoints: {len(keypoint_indices)}")
    
    # Setup rendering
    if show_window:
        window = init_glfw_window(800, 600)
        width, height = glfw.get_framebuffer_size(window)
    else:
        width, height = 800, 600  # Maximum for default offscreen buffer
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Setup camera
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = np.array([0.0, 1.2, 0.4])
    camera.distance = 2.5
    camera.elevation = -20
    camera.azimuth = 135
    
    print(f"\n🎬 Starting rendering...")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total actions: {len(trajectories)}")
    print(f"   Show window: {show_window}")
    
    frames = []
    frame_count = 0
    
    # Add goal visualization spheres to model
    goal_geom_ids = []
    
    try:
        # Render each action
        for action_idx, traj in enumerate(trajectories):
            print(f"\n🎯 Rendering action {action_idx + 1}/{len(trajectories)}")
            
            n_steps = len(traj)
            print(f"   Steps: {n_steps}")
            
            # Subsample for reasonable video length
            step_skip = max(1, n_steps // 50)  # ~50 frames per action
            
            for step_idx in range(0, n_steps, step_skip):
                # Set cloth keypoint positions
                keypoint_positions = traj[step_idx]  # Shape: (9, 3)
                
                for kp_idx, body_id in enumerate(keypoint_indices):
                    if body_id >= 0 and kp_idx < len(keypoint_positions):
                        # Set body position
                        data_mj.xpos[body_id] = keypoint_positions[kp_idx]
                
                # Forward kinematics to update visualization
                mujoco.mj_forward(model, data_mj)
                
                # Render frame
                renderer.update_scene(data_mj, camera=camera)
                pixels = renderer.render()
                
                # Draw goal markers (red spheres) on the image
                # This would require additional overlay rendering
                
                frames.append(pixels.copy())
                frame_count += 1
                
                # Update window if showing
                if show_window and window:
                    # Copy pixels to OpenGL
                    import OpenGL.GL as gl
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                    gl.glDrawPixels(width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    
                    if glfw.window_should_close(window):
                        print("\n⚠️  Window closed by user")
                        break
                
                if frame_count % 100 == 0:
                    print(f"   Rendered {frame_count} frames...")
            
            if show_window and window and glfw.window_should_close(window):
                break
    
    finally:
        # Cleanup
        if show_window and window:
            glfw.destroy_window(window)
            glfw.terminate()
        
        temp_xml.unlink(missing_ok=True)
    
    print(f"\n✅ Rendering complete!")
    print(f"   Total frames: {len(frames)}")
    
    # Save video
    if output_video is None:
        output_video = pathlib.Path(data_path).parent / f"cloth_mujoco_{episode_id}.mp4"
    output_video = str(output_video)
    
    if len(frames) > 0:
        print(f"\n💾 Saving video...")
        print(f"   Output: {output_video}")
        
        fps = 30
        imageio.mimsave(output_video, frames, fps=fps, quality=9, macro_block_size=1)
        
        print(f"✅ Video saved!")
        print(f"   Duration: {len(frames) / fps:.1f}s")
        print(f"   FPS: {fps}")
    else:
        print(f"⚠️  No frames rendered!")
    
    return output_video


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_cloth_mujoco.py <trajectory_data.pkl> [output.mp4] [--no-window]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    show_window = '--no-window' not in sys.argv
    
    try:
        render_cloth_trajectory(data_path, output_video, show_window)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
