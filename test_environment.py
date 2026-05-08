#!/usr/bin/env python3
"""
Test script to verify environment setup without replay
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import numpy as np

def test_environment():
    """Test basic environment functionality"""
    
    print("=== Testing TableClothSimEnvironment ===\n")
    
    # Setup
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
    
    print("Creating environment (no visualization)...")
    env = TableClothSimEnvironment(
        rope_config, 
        controller_config,
        obs_topdown=False,
        show_vis=False
    )
    print("✓ Environment created\n")
    
    # Test goal setting
    print("Setting goal (alpha=0.5)...")
    goal = env.get_cloth_goal(0.5)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    print("✓ Goal set\n")
    
    # Test a few steps with simple actions
    print("Testing 5 steps with init action...")
    init_action = np.array([0.87, 0.8, 0.7, 0.3])
    
    for i in range(5):
        try:
            observation, loss, _, info = env.step(init_action)
            print(f"  Step {i}: loss = {loss:.6f}")
        except Exception as e:
            print(f"  Step {i}: ERROR - {e}")
            break
    
    print("\n✓ Environment test completed!")

if __name__ == "__main__":
    test_environment()
