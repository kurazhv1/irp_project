#!/usr/bin/env python3
"""
Test ONLY first step with visualization to isolate when segfault occurs
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import numpy as np
import time

def test_single_step_with_vis():
    """
    Test with just ONE step and visualization
    This will help identify exactly when segfault happens
    """
    print("=== Single Step Visualization Test ===\n")
    
    rope_config = {
        'table_height': 0.8,
        'table_y': 1,
        'table_size': 1.2,
        'cloth_spacing': 0.0383,
        'cloth_density': 0.98,
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print("Step 1: Creating environment WITH visualization...")
    try:
        env = TableClothSimEnvironment(
            rope_config, 
            controller_config,
            obs_topdown=False,
            show_vis=True  # ENABLE VISUALIZATION
        )
        print("✓ Environment created successfully!")
        print("   Viewer window should be visible now")
    except Exception as e:
        print(f"✗ FAILED at environment creation: {e}")
        return
    
    print("\nStep 2: Setting goal...")
    try:
        goal = env.get_cloth_goal(0.5)
        loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
        env.set_loss_func(loss_func)
        print("✓ Goal set")
    except Exception as e:
        print(f"✗ FAILED at goal setting: {e}")
        return
    
    print("\nStep 3: Waiting 2 seconds to see initial state...")
    time.sleep(2)
    print("✓ Initial visualization visible")
    
    print("\nStep 4: Executing ONE action step...")
    action = np.array([0.87, 0.8, 0.7, 0.3])
    print(f"   Action: {action}")
    
    try:
        observation, loss, _, info = env.step(action)
        print(f"✓ Action executed successfully!")
        print(f"   Loss: {loss:.6f}")
    except Exception as e:
        print(f"✗ FAILED during action execution: {e}")
        return
    
    print("\nStep 5: Waiting 3 seconds to see result...")
    time.sleep(3)
    
    print("\n✓ Test completed without segfault!")
    print("Press Ctrl+C to close...")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nClosing...")

if __name__ == "__main__":
    print("=" * 60)
    print("WARNING: This will open MuJoCo viewer window")
    print("If segfault occurs, we'll know at which step it happens")
    print("=" * 60)
    input("\nPress Enter to start test...")
    
    test_single_step_with_vis()
