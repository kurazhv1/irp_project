#!/usr/bin/env python3
"""
Simple replay script using the actual TableClothSimEnvironment
Works with mujoco-py (legacy environment)
"""

import json
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import sys

# Add project to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from environments.table_cloth_sim_environment import TableClothSimEnvironment

def load_action_log(log_path):
    """Load action log from JSON file"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data

def replay_actions_simple(action_log, show_vis=False):
    """
    Replay action sequence using the real environment
    
    Args:
        action_log: Loaded action log dictionary
        show_vis: Whether to show visualization (may cause segfault)
    """
    metadata = action_log['metadata']
    actions = action_log['actions']
    
    print("\n=== Metadata ===")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n=== Replay Info ===")
    print(f"Total steps: {len(actions)}")
    print(f"Rope params: {metadata['rope_param']}")
    print(f"Goal alpha: {metadata['goal_alpha']}")
    
    # Setup environment with the same parameters as in the log
    rope_config = {
        'table_height': 0.8,
        'table_y': 1,
        'table_size': 1.2,
        'cloth_spacing': metadata['rope_param'][0] / 12,
        'cloth_density': metadata['rope_param'][1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print(f"\n=== Creating environment ===")
    print(f"Cloth spacing: {rope_config['cloth_spacing']:.4f}")
    print(f"Cloth density: {rope_config['cloth_density']:.2f}")
    print(f"Show visualization: {show_vis}")
    
    try:
        env = TableClothSimEnvironment(
            rope_config, 
            controller_config,
            obs_topdown=False,
            show_vis=show_vis  # Use parameter to control visualization
        )
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return None
    
    # Get goal
    goal = env.get_cloth_goal(metadata['goal_alpha'])
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0,1,2])
    env.set_loss_func(loss_func)
    
    print(f"\n=== Replaying {len(actions)} actions ===")
    
    errors = []
    for action_step in tqdm(actions, desc="Replaying"):
        action = np.array(action_step['action'])
        
        try:
            # Step environment
            observation, loss, _, info = env.step(action)
            errors.append(loss)
            
            # Compare with logged loss
            logged_loss = action_step['loss']
            loss_diff = abs(loss - logged_loss)
            
            if loss_diff > 0.01:  # Significant difference
                print(f"\nWarning at step {action_step['step_id']}:")
                print(f"  Logged loss: {logged_loss:.6f}")
                print(f"  Replay loss: {loss:.6f}")
                print(f"  Difference: {loss_diff:.6f}")
                
        except Exception as e:
            print(f"\nError at step {action_step['step_id']}: {e}")
            break
    
    print(f"\n=== Results ===")
    print(f"Final error: {errors[-1]:.6f}")
    print(f"Min error: {min(errors):.6f}")
    print(f"Max error: {max(errors):.6f}")
    
    # Compare with logged min error
    logged_errors = [a['loss'] for a in actions]
    print(f"\nLogged min error: {min(logged_errors):.6f}")
    print(f"Replay min error: {min(errors):.6f}")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Replay action logs using real environment")
    parser.add_argument("log_file", type=str, help="Path to action_log JSON file")
    parser.add_argument("--show-vis", action="store_true", help="Show visualization (may cause segfault)")
    
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Action log not found: {log_path}")
    
    print(f"Loading action log: {log_path}")
    action_log = load_action_log(log_path)
    
    # Replay
    try:
        errors = replay_actions_simple(action_log, show_vis=args.show_vis)
        
        if errors:
            print("\n✓ Replay completed successfully!")
        else:
            print("\n✗ Replay failed")
            
    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user")
    except Exception as e:
        print(f"\n✗ Replay error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
