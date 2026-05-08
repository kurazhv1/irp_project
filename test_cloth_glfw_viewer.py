#!/usr/bin/env python3
"""
Cloth visualization using modern MuJoCo Python API with GLFW
Direct rendering without mujoco-py viewer wrapper
"""

import mujoco
import glfw
import numpy as np
from pathlib import Path

def main():
    xml_path = Path(__file__).parent / "assets/mujoco/cloth/cloth.xml"
    
    print("=" * 60)
    print("  CLOTH GLFW VIEWER TEST")
    print("=" * 60)
    print(f"📂 Loading model: {xml_path}")
    
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Actuators: {model.nu}")

    if not glfw.init():
        raise Exception("❌ GLFW init failed")
    print("✅ GLFW initialized")

    window = glfw.create_window(1200, 900, "Cloth Test Viewer", None, None)
    if not window:
        glfw.terminate()
        raise Exception("❌ Failed to create GLFW window")
    
    glfw.make_context_current(window)
    print("✅ GLFW window created")

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    print("✅ MuJoCo rendering context created")

    # Initial camera setup
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 90
    cam.elevation = -30
    cam.distance = 2.5
    cam.lookat[:] = [0.5, 0.8, 0.3]
    
    print("\n🎬 Starting visualization...")
    print("   Controls:")
    print("   - Random gripper movements")
    print("   - Close window or press ESC to exit\n")

    step_count = 0
    target_change_interval = 50
    
    try:
        while not glfw.window_should_close(window):
            # Random gripper actions
            if step_count % target_change_interval == 0:
                gy = np.random.uniform(-0.1, 0.1)
                gz = np.random.uniform(-0.1, 0.1)
                print(f"Step {step_count}: New gripper control: gy={gy:.3f}, gz={gz:.3f}")
            
            data.ctrl[:] = [gy, gz]

            # Physics step
            mujoco.mj_step(model, data)

            # Render
            w, h = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, w, h)

            mujoco.mjv_updateScene(model, data, opt, None, cam, 
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)

            glfw.swap_buffers(window)
            glfw.poll_events()
            
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
