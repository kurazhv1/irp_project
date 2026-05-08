#!/usr/bin/env python3
"""
Test cloth visualization using sim.render() instead of viewer.

This avoids the viewer.render() segfault by using sim.render() directly.

Usage:
    conda activate irp_legacy
    python test_cloth_sim_render.py
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import mujoco_py as mj


def test_sim_render():
    """Test cloth simulation with sim.render() instead of viewer."""
    
    print("=" * 70)
    print("🧪 TEST SIM.RENDER() FOR CLOTH VISUALIZATION")
    print("=" * 70)
    
    # Create environment WITHOUT visualization
    print("\n📋 Creating environment (headless)...")
    
    rope_config = dict(
        table_height=0.8,
        table_y=1,
        table_size=1.2,
        cloth_spacing=0.05,
        cloth_density=1.4
    )
    
    controller_config = dict(
        joint_names=['gy', 'gz'],
        kp=100000,
        kv=100000
    )
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        show_vis=False,  # NO viewer!
        obs_topdown=False
    )
    print("   ✅ Environment created")
    
    # Setup goal and loss
    goal = env.get_cloth_goal(0.0)
    loss_func = env.get_traj_loss_func(goal, n_step=5, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    print("\n📋 Testing sim.render() for visualization...")
    print("   Trying to render frame using sim.render()...")
    
    try:
        # Try to render a frame
        width = 640
        height = 480
        
        # Method 1: Try sim.render()
        try:
            frame = env.sim.render(width, height, camera_name=None)
            print(f"   ✅ sim.render() works! Frame shape: {frame.shape}")
            
            # Save frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test_frame_initial.png', frame_bgr)
            print(f"   ✅ Saved: test_frame_initial.png")
            
        except Exception as e:
            print(f"   ⚠️  sim.render() failed: {e}")
            print("   Trying alternative methods...")
            
            # Method 2: Try with camera_id
            try:
                frame = env.sim.render(width, height, camera_id=0)
                print(f"   ✅ sim.render(camera_id=0) works! Frame shape: {frame.shape}")
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite('test_frame_initial.png', frame_bgr)
                print(f"   ✅ Saved: test_frame_initial.png")
                
            except Exception as e2:
                print(f"   ❌ sim.render(camera_id=0) also failed: {e2}")
                return False
        
        # Now take a step and render again
        print("\n📋 Taking action and rendering...")
        action = np.array([0.8, 0.7, 0.6, 0.3])
        print(f"   Action: {action}")
        
        obs, loss, done, info = env.step(action)
        print(f"   ✅ Step successful! Loss: {loss:.6f}")
        
        # Render after step
        frame = env.sim.render(width, height)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test_frame_after_step.png', frame_bgr)
        print(f"   ✅ Saved: test_frame_after_step.png")
        
        # Take a few more steps and save frames
        print("\n📋 Taking 4 more steps and rendering each...")
        for i in range(4):
            action = np.random.uniform([0.6, 0.5, 0.5, 0.2], [0.9, 0.9, 0.8, 0.4])
            obs, loss, done, info = env.step(action)
            
            frame = env.sim.render(width, height)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'test_frame_step_{i+2}.png', frame_bgr)
            print(f"   Step {i+2}: loss={loss:.6f}, saved frame")
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! sim.render() works without segfault!")
        print("=" * 70)
        print(f"\n📸 Saved frames:")
        print(f"   - test_frame_initial.png")
        print(f"   - test_frame_after_step.png")
        print(f"   - test_frame_step_2.png")
        print(f"   - test_frame_step_3.png")
        print(f"   - test_frame_step_4.png")
        print(f"   - test_frame_step_5.png")
        print("\n💡 Check these PNG files to see the cloth simulation!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = test_sim_render()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
