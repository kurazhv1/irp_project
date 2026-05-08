#!/usr/bin/env python3
"""
Test cloth visualization with random gripper movements.

This script:
1. Loads cloth.xml with full cloth physics
2. Applies random gripper actions
3. Visualizes in MuJoCo viewer (mujoco-py)

This proves that cloth visualization WORKS and can be used for diploma screenshots.

Usage:
    conda activate irp_legacy
    python test_cloth_random_viz.py

Controls:
    - Mouse: Rotate/pan view
    - Space: Pause/resume
    - Backspace: Reset
    - Ctrl+C: Exit
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import mujoco_py as mj
except ImportError:
    print("❌ Error: mujoco-py not available!")
    print("   Run: conda activate irp_legacy")
    sys.exit(1)


def test_cloth_visualization_random():
    """
    Test cloth visualization with random gripper movements.
    """
    print("=" * 70)
    print("🎬 CLOTH VISUALIZATION TEST - RANDOM ACTIONS")
    print("=" * 70)
    
    # Load cloth.xml
    xml_path = project_root / "assets" / "mujoco" / "cloth" / "cloth.xml"
    
    if not xml_path.exists():
        print(f"❌ Error: cloth.xml not found: {xml_path}")
        sys.exit(1)
    
    print(f"\n📂 Loading: {xml_path}")
    
    try:
        model = mj.load_model_from_path(str(xml_path))
        sim = mj.MjSim(model)
        viewer = mj.MjViewer(sim)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    print("✅ Model loaded successfully!")
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Actuators: {model.nu}")
    
    # Find gripper body
    gripper_body_id = None
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and 'gripper' in body_name.lower():
            gripper_body_id = i
            print(f"\n🤖 Found gripper body: '{body_name}' (id={i})")
            break
    
    if gripper_body_id is None:
        print("\n⚠️  No gripper body found, will just run simulation")
    
    # Find cloth bodies
    cloth_body_ids = []
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and ('B0' in body_name or 'cloth' in body_name.lower()):
            cloth_body_ids.append(i)
    
    print(f"🧵 Found {len(cloth_body_ids)} cloth bodies")
    
    print("\n⌨️  Controls:")
    print("   - Mouse drag: Rotate view")
    print("   - Mouse scroll: Zoom")
    print("   - Space: Pause/Resume")
    print("   - Backspace: Reset")
    print("   - Ctrl+C: Exit")
    print("=" * 70)
    print("\n▶️  Starting visualization with random gripper movements...\n")
    
    # Simulation parameters
    step_count = 0
    max_steps = 10000  # Run for a long time
    action_change_interval = 100  # Change action every N steps
    
    # Random action state
    target_pos = np.array([0.0, 1.0, 0.3])  # Initial gripper position
    velocity = np.zeros(3)
    
    try:
        while step_count < max_steps:
            # Change target position periodically
            if step_count % action_change_interval == 0:
                # Random target in workspace
                target_pos = np.array([
                    np.random.uniform(-0.4, 0.4),   # x: side to side
                    np.random.uniform(0.5, 1.5),    # y: front/back
                    np.random.uniform(0.0, 0.5)     # z: height
                ])
                print(f"Step {step_count:5d}: New target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
            
            # Move gripper towards target (if we have gripper)
            if gripper_body_id is not None:
                # Get current gripper position
                current_pos = sim.data.body_xpos[gripper_body_id].copy()
                
                # Simple PD controller
                error = target_pos - current_pos
                velocity = error * 0.1  # Proportional gain
                
                # Apply velocity (by setting qvel if gripper has joints)
                # For freejoint: qvel indices 0-2 are linear velocity
                if model.body_jntadr[gripper_body_id] >= 0:
                    jnt_addr = model.body_jntadr[gripper_body_id]
                    jnt_type = model.jnt_type[jnt_addr]
                    
                    if jnt_type == 0:  # Free joint
                        qvel_addr = model.jnt_qposadr[jnt_addr]
                        sim.data.qvel[qvel_addr:qvel_addr+3] = velocity
            
            # Step simulation
            sim.step()
            
            # Render
            viewer.render()
            
            step_count += 1
            
            # Small delay to make it visible
            time.sleep(0.001)
            
            # Check if viewer is still open
            if not viewer._running:
                print("\n👋 Viewer closed")
                break
        
        if step_count >= max_steps:
            print(f"\n✅ Completed {max_steps} steps")
            print("   Close viewer window or press Ctrl+C to exit")
            
            # Keep viewer open
            while viewer._running:
                viewer.render()
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🎉 DONE!")
    print("=" * 70)


def main():
    # Check if we're in the right environment
    try:
        import mujoco_py
        print("✅ mujoco-py available")
    except ImportError:
        print("❌ Error: mujoco-py not available!")
        print("\nPlease activate the correct environment:")
        print("   conda activate irp_legacy")
        sys.exit(1)
    
    test_cloth_visualization_random()


if __name__ == '__main__':
    main()
