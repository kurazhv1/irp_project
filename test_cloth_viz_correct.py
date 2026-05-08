#!/usr/bin/env python3
"""
Test cloth visualization with random gripper movements
Using CORRECT approach: viewer.render() AFTER sim.step()
"""

import numpy as np
import mujoco_py as mj
import time
from pathlib import Path

def main():
    # Load model
    xml_path = Path(__file__).parent / 'assets' / 'mujoco' / 'cloth' / 'cloth.xml'
    print(f"📂 Loading model: {xml_path}")
    
    model = mj.load_model_from_path(str(xml_path))
    sim = mj.MjSim(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Bodies: {model.nbody}")
    print(f"   Joints: {model.njnt}")
    print(f"   Actuators: {model.nu}")
    
    # Find gripper body
    gripper_body_id = None
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and 'gripper' in body_name.lower():
            gripper_body_id = i
            print(f"🤖 Found gripper body: '{body_name}' (id={i})")
            break
    
    if gripper_body_id is None:
        print("❌ Gripper body not found!")
        return
    
    # Find cloth bodies
    cloth_body_ids = []
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and 'B' in body_name and '_' in body_name:
            cloth_body_ids.append(i)
    print(f"🧵 Found {len(cloth_body_ids)} cloth bodies")
    
    # Create viewer AFTER sim is ready
    print("🖼️  Creating viewer...")
    sim.forward()  # ВАЖНО: forward() перед созданием viewer
    viewer = mj.MjViewer(sim)
    print("✅ Viewer created!")
    
    # Random motion parameters
    np.random.seed(42)
    motion_range = {
        'x': (0.0, 0.5),
        'y': (0.3, 1.2),
        'z': (0.2, 0.6)
    }
    
    target_pos = None
    steps_to_target = 0
    step_count = 0
    max_steps = 2000
    
    print("\n🎬 Starting visualization...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while step_count < max_steps:
            # Generate new random target periodically
            if steps_to_target <= 0:
                target_pos = np.array([
                    np.random.uniform(*motion_range['x']),
                    np.random.uniform(*motion_range['y']),
                    np.random.uniform(*motion_range['z'])
                ])
                steps_to_target = np.random.randint(50, 150)
                print(f"Step {step_count}: New target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
            
            # Get current gripper position
            current_pos = sim.data.body_xpos[gripper_body_id].copy()
            
            # Smooth motion towards target
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.01:
                # Apply small force in direction of target
                force_magnitude = 5.0  # Much smaller force
                force = (direction / distance) * force_magnitude
                sim.data.xfrc_applied[gripper_body_id, :3] = force
            else:
                sim.data.xfrc_applied[gripper_body_id, :3] = 0
            
            # CORRECT ORDER: sim.step() first, then viewer.render()
            sim.step()
            viewer.render()
            
            steps_to_target -= 1
            step_count += 1
            
            # Small delay for visualization
            time.sleep(0.002)
        
        print(f"\n✅ Visualization completed! ({step_count} steps)")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Closing viewer...")

if __name__ == "__main__":
    print("=" * 60)
    print("  CLOTH VISUALIZATION TEST (Correct Approach)")
    print("=" * 60)
    main()
