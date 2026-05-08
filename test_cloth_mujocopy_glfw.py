#!/usr/bin/env python3
"""
Cloth visualization using mujoco-py with manual GLFW rendering
Bypassing MjViewer wrapper to avoid segfaults
"""

import mujoco_py as mj
import glfw
from OpenGL.GL import *
import numpy as np
from pathlib import Path

def main():
    xml_path = Path(__file__).parent / "assets/mujoco/cloth/cloth.xml"
    
    print("=" * 60)
    print("  CLOTH MUJOCO-PY + GLFW VIEWER")
    print("=" * 60)
    print(f"📂 Loading model: {xml_path}")
    
    # Load model with mujoco-py (supports old cloth format)
    model = mj.load_model_from_path(str(xml_path))
    sim = mj.MjSim(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Actuators: {model.nu}")

    # Initialize simulation
    sim.forward()
    
    # Find gripper body
    gripper_body_id = None
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and 'gripper' in body_name.lower():
            gripper_body_id = i
            print(f"🤖 Found gripper body: '{body_name}' (id={i})")
            break

    if not glfw.init():
        raise Exception("❌ GLFW init failed")
    print("✅ GLFW initialized")

    window = glfw.create_window(1200, 900, "Cloth mujoco-py Viewer", None, None)
    if not window:
        glfw.terminate()
        raise Exception("❌ Failed to create GLFW window")
    
    glfw.make_context_current(window)
    print("✅ GLFW window created")

    # Setup offscreen renderer from mujoco-py
    render_context = mj.MjRenderContextOffscreen(sim, 0)
    
    print("✅ Render context created")
    
    print("\n🎬 Starting visualization...")
    print("   Controls:")
    print("   - Random gripper movements")
    print("   - Close window or press ESC to exit\n")

    step_count = 0
    target_pos = None
    steps_to_target = 0
    
    motion_range = {
        'x': (0.0, 0.5),
        'y': (0.3, 1.2),
        'z': (0.2, 0.6)
    }
    
    try:
        while not glfw.window_should_close(window):
            # Generate new random target periodically
            if steps_to_target <= 0:
                target_pos = np.array([
                    np.random.uniform(*motion_range['x']),
                    np.random.uniform(*motion_range['y']),
                    np.random.uniform(*motion_range['z'])
                ])
                steps_to_target = np.random.randint(50, 150)
                print(f"Step {step_count}: New target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
            
            if gripper_body_id is not None:
                # Get current gripper position
                current_pos = sim.data.body_xpos[gripper_body_id].copy()
                
                # Smooth motion towards target
                direction = target_pos - current_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0.01:
                    # Apply small force in direction of target
                    force_magnitude = 5.0
                    force = (direction / distance) * force_magnitude
                    sim.data.xfrc_applied[gripper_body_id, :3] = force
                else:
                    sim.data.xfrc_applied[gripper_body_id, :3] = 0
            
            # Physics step
            sim.step()
            
            # Render using offscreen context (-1 = free camera)
            render_context.render(1200, 900, -1)
            img = render_context.read_pixels(1200, 900, depth=False)
            img = img[::-1, :, :]  # Flip vertically
            
            # Display in GLFW window
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDrawPixels(1200, 900, GL_RGB, GL_UNSIGNED_BYTE, img)
            
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            steps_to_target -= 1
            step_count += 1
            
            # Exit on ESC
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                print("\n⚠️  ESC pressed, exiting...")
                break
        
        print(f"\n✅ Visualization completed! ({step_count} steps)")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Cleaning up...")
        glfw.terminate()

if __name__ == "__main__":
    main()
