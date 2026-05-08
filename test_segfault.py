#!/usr/bin/env python3
"""
Minimal test to isolate segfault issue
Uses hardcoded actions and simple visualization
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment
import numpy as np
import time

def test_with_hardcoded_actions(show_vis=False):
    """
    Test with simple hardcoded actions
    """
    print("=== Minimal Segfault Test ===\n")
    
    # Simple rope config
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
    
    # Hardcoded actions (from our test log)
    actions = [
        [0.87, 0.8, 0.7, 0.3],  # init action
        [0.74, 0.77, 0.68, 0.55],  # action from step 1
        [0.67, 0.80, 0.72, 0.80],  # action from step 2
    ]
    
    print(f"Creating environment (show_vis={show_vis})...")
    try:
        env = TableClothSimEnvironment(
            rope_config, 
            controller_config,
            obs_topdown=False,
            show_vis=show_vis
        )
        print("✓ Environment created\n")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Set goal
    print("Setting goal...")
    goal = env.get_cloth_goal(0.5)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    print("✓ Goal set\n")
    
    # Execute actions
    print(f"Executing {len(actions)} hardcoded actions...")
    for i, action in enumerate(actions):
        print(f"\nStep {i}:")
        print(f"  Action: {action}")
        
        try:
            observation, loss, _, info = env.step(action)
            print(f"  Loss: {loss:.6f}")
            print(f"  ✓ Step completed")
            
            if show_vis:
                time.sleep(0.5)  # Pause to see visualization
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False
    
    print("\n✓ All steps completed successfully!")
    
    if show_vis:
        print("\nViewer is open. Press Ctrl+C to close...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nClosing...")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-vis", action="store_true", help="Show visualization (may segfault)")
    args = parser.parse_args()
    
    print("=" * 50)
    if args.show_vis:
        print("WARNING: Running with visualization - may cause segfault!")
        print("=" * 50)
        input("Press Enter to continue or Ctrl+C to abort...")
    
    success = test_with_hardcoded_actions(show_vis=args.show_vis)
    
    if success:
        print("\n🎉 Test passed!")
    else:
        print("\n❌ Test failed!")
