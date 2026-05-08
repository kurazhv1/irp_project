#!/usr/bin/env python3
"""
Minimal test of cloth visualization with random actions.

This is a step-by-step test to isolate the segfault issue:
1. Create environment with show_vis=True
2. Take random actions
3. See if visualization works

Usage:
    conda activate irp_legacy
    python test_cloth_vis_minimal.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def test_minimal_visualization():
    """Test minimal cloth visualization with random actions."""
    
    print("=" * 70)
    print("🧪 MINIMAL CLOTH VISUALIZATION TEST")
    print("=" * 70)
    
    # Step 1: Create environment
    print("\n📋 Step 1: Creating environment WITHOUT visualization...")
    
    rope_config = dict(
        table_height=0.8,
        table_y=1,
        table_size=1.2,
        cloth_spacing=0.05,  # Default value
        cloth_density=1.4     # Default value
    )
    
    controller_config = dict(
        joint_names=['gy', 'gz'],
        kp=100000,
        kv=100000
    )
    
    print("   Creating TableClothSimEnvironment(show_vis=False)...")
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        show_vis=False,  # Start without viz
        obs_topdown=False
    )
    print("   ✅ Environment created successfully (headless)")
    
    # Step 2: Test headless simulation
    print("\n📋 Step 2: Testing headless simulation with random action...")
    
    # Setup goal and loss
    goal = env.get_cloth_goal(0.0)
    loss_func = env.get_traj_loss_func(goal, n_step=5, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    # Random action
    action = np.array([0.8, 0.7, 0.6, 0.3])
    print(f"   Action: {action}")
    
    try:
        obs, loss, done, info = env.step(action)
        print(f"   ✅ Step successful! Loss: {loss:.6f}")
        print(f"   Observation shape: {obs.shape}")
    except Exception as e:
        print(f"   ❌ Step failed: {type(e).__name__}: {e}")
        return False
    
    # Step 3: Try with visualization
    print("\n📋 Step 3: Creating NEW environment WITH visualization...")
    print("   ⚠️  This may cause segfault!")
    
    input("\nPress ENTER to create environment with show_vis=True...")
    
    print("   Creating TableClothSimEnvironment(show_vis=True)...")
    try:
        env_vis = TableClothSimEnvironment(
            rope_config=rope_config,
            controller_config=controller_config,
            show_vis=True,  # Enable visualization
            obs_topdown=False
        )
        print("   ✅ Environment with visualization created!")
        print("   ✅ Window should be visible now")
        
        # Setup goal and loss for new env
        goal_vis = env_vis.get_cloth_goal(0.0)
        loss_func_vis = env_vis.get_traj_loss_func(goal_vis, n_step=5, measure_dims=[0,1,2])
        env_vis.set_loss_func(loss_func_vis)
        
        # Step 4: Try to step with visualization
        print("\n📋 Step 4: Taking action WITH visualization enabled...")
        print("   ⚠️  This is where segfault usually happens!")
        
        input("\nPress ENTER to call env.step() with visualization...")
        
        print(f"   Calling env.step({action})...")
        obs_vis, loss_vis, done_vis, info_vis = env_vis.step(action)
        
        print(f"   ✅ SUCCESS! Step with visualization worked!")
        print(f"   Loss: {loss_vis:.6f}")
        
        # Try a few more steps
        print("\n📋 Step 5: Taking 4 more steps...")
        for i in range(4):
            # Random action
            action = np.random.uniform([0.6, 0.5, 0.5, 0.2], [0.9, 0.9, 0.8, 0.4])
            print(f"   Step {i+2}: action={action[:2]}")
            obs_vis, loss_vis, done_vis, info_vis = env_vis.step(action)
            print(f"      → loss={loss_vis:.6f}")
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n💡 Visualization window should still be open.")
        print("   Close window or press Ctrl+C to exit.")
        
        # Keep window open
        input("\nPress ENTER to close...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = test_minimal_visualization()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
