#!/usr/bin/env python3
"""
Test: Create with viewer, but close it before step
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import numpy as np

def test_close_viewer_before_step():
    """
    Create environment with viewer, see initial state, then close viewer and continue
    """
    print("=== Test: Close Viewer Before Step ===\n")
    
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
    
    print("Creating environment WITH visualization...")
    env = TableClothSimEnvironment(
        rope_config, 
        controller_config,
        obs_topdown=False,
        show_vis=True
    )
    print("✓ Environment created with viewer\n")
    
    print("Setting goal...")
    goal = env.get_cloth_goal(0.5)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    print("✓ Goal set\n")
    
    print("Closing viewer before stepping...")
    if hasattr(env, 'viewer') and env.viewer is not None:
        # Close the viewer
        del env.viewer
        env.viewer = None
        print("✓ Viewer closed\n")
    
    print("Now executing action WITHOUT viewer...")
    action = np.array([0.87, 0.8, 0.7, 0.3])
    
    try:
        observation, loss, _, info = env.step(action)
        print(f"✓ Action executed!")
        print(f"   Loss: {loss:.6f}")
        print("\n✅ Success! No segfault when viewer is closed before step")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_close_viewer_before_step()
