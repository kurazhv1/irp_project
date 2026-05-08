#!/usr/bin/env python3
"""
Create test action log with random actions for visualization testing

Simulates one episode and saves action log for replay_cloth_trained_model.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment


def create_test_action_log():
    """Create a test action log with random actions"""
    
    # Rope config
    rope_config = {
        'table_height': 0.8,
        'table_y': 1,
        'table_size': 1.2,
        'cloth_spacing': 0.05,
        'cloth_density': 1.4
    }
    
    # Controller config
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print("=" * 70)
    print("  CREATE TEST ACTION LOG")
    print("=" * 70)
    print()
    
    # Create environment
    print("🔧 Creating environment...")
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        dt=0.01,
        max_steps=400,
        show_vis=False,  # Headless
        obs_topdown=False
    )
    
    # Set goal
    goal_alpha = 0.5
    goal_coords = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal_coords)
    env.set_loss_func(loss_func)
    
    print(f"✅ Environment created")
    print(f"   Goal alpha: {goal_alpha}")
    print(f"   Goal coords shape: {goal_coords.shape}")
    
    # Generate random actions
    np.random.seed(42)
    n_actions = 5
    actions = []
    raw_actions = []
    
    print(f"\n🎲 Generating {n_actions} random actions...")
    
    for i in range(n_actions):
        # Random normalized action [0,1]
        action = np.random.rand(4).astype(np.float32)
        actions.append(action.tolist())
        
        # Map to raw action (duration, gy1, gz1, gy2)
        raw_action = env.action_mapper(action)
        raw_actions.append(raw_action.tolist())
        
        print(f"  Action {i}: {action} → raw: {raw_action}")
    
    # Run simulation
    print(f"\n🏃 Running simulation...")
    
    # Use first action for actual simulation
    obs, loss, done, info = env.step(actions[0])
    trajectory = info['trajectory']
    trajectory_pix = info['trajectory_pix']
    
    print(f"✅ Simulation complete")
    print(f"   Loss: {loss:.6f}")
    print(f"   Trajectory shape: {trajectory.shape}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / 'output' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create action log
    action_log = {
        'episode_id': 'test_rope0_goal_0.5',
        'timestamp': timestamp,
        'rope_config': rope_config,
        'controller_config': controller_config,
        'goal_alpha': goal_alpha,
        'goal_coords': goal_coords.tolist(),
        'actions': actions,
        'raw_actions': raw_actions,
        'loss': float(loss),
        'trajectory': trajectory.tolist(),
        'trajectory_pix': trajectory_pix.tolist(),
        'observation_shape': list(obs.shape),
        'info': {
            'n_actions': n_actions,
            'test_data': True,
            'note': 'Random actions for visualization testing'
        }
    }
    
    # Save action log
    log_path = output_dir / 'action_log_test_random.json'
    with open(log_path, 'w') as f:
        json.dump(action_log, f, indent=2)
    
    print(f"\n💾 Action log saved:")
    print(f"   {log_path}")
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    metadata = {
        'timestamp': timestamp,
        'n_episodes': 1,
        'rope_configs': [rope_config],
        'goal_alphas': [goal_alpha],
        'test_mode': True
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   {metadata_path}")
    
    print(f"\n✅ Test action log created successfully!")
    print(f"\n📝 Next step:")
    print(f"   python replay_cloth_trained_model.py {log_path}")
    print()
    
    return log_path


if __name__ == "__main__":
    log_path = create_test_action_log()
