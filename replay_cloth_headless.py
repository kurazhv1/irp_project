#!/usr/bin/env python
"""
Cloth simulation replay using ORIGINAL mujoco-py environment - HEADLESS
Saves keypoint trajectories without rendering (works everywhere!)
Then creates visualization from saved data.
"""

import sys
import json
import pathlib
import numpy as np
import pickle
from scipy.interpolate import CubicSpline

# Import original environment components
from environments.table_cloth_sim_environment import (
    TableClothSimEnvironment, 
    get_cubic_control
)
from abr_control_mod.mujoco_utils import get_body_center_of_mass


def replay_episode_headless(action_log_path: str, output_data: str = None):
    """
    Replay episode and save keypoint trajectories (no rendering!)
    
    Args:
        action_log_path: Path to action log JSON
        output_data: Output pickle path (default: same name as log with .pkl)
    """
    
    # Load action log
    with open(action_log_path, 'r') as f:
        log_data = json.load(f)
    
    episode_id = log_data['episode_id']
    actions = log_data['actions']
    goal_coords = np.array(log_data['goal_coords'])
    metadata = log_data['metadata']
    logged_loss = log_data['loss']
    
    print(f"📄 Loaded action log: {pathlib.Path(action_log_path).name}")
    print(f"   Episode: {episode_id}")
    print(f"   Logged loss: {logged_loss:.6f} ({logged_loss * 100:.2f}cm)")
    print(f"   Actions: {len(actions)}")
    
    # Create environment (headless, no viewer, no renderer!)
    rope_param = metadata['rope_param']
    rope_config = {
        'table_height': 0.8,
        'table_y': 1.0,
        'table_size': 1.2,
        'cloth_spacing': rope_param[0] / 12,
        'cloth_density': rope_param[1],
    }
    
    controller_config = {
        'joint_names': ['gy', 'gz'],
        'kp': 100000,
        'kv': 100000
    }
    
    print(f"🏗️  Creating environment (headless)...")
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        dt=0.01,
        max_steps=400,
        show_vis=False  # NO VIEWER!
    )
    
    sim = env.sim
    
    print(f"🎬 Replaying episode...")
    print(f"   Using ORIGINAL TableClothSimEnvironment")
    print(f"   Controller: MujocoCompensatedPDController (kp={controller_config['kp']}, kv={controller_config['kv']})")
    
    # Store all data
    all_trajectories = []
    all_losses = []
    all_actions_info = []
    
    # Reset to initial state ONCE at the start
    print(f"🔄 Resetting to initial state...")
    env.ctrl._load_state(*env.init_state)
    
    # Replay each action SEQUENTIALLY (each builds on previous!)
    for action_idx, action in enumerate(actions):
        print(f"\n🎯 Action {action_idx + 1}/{len(actions)}")
        
        # NO RESET HERE! Each action continues from previous state!
        
        # Generate trajectory
        raw_action = env.action_mapper(np.array(action))
        duration, gy1, gz1, gy2 = raw_action
        gz2 = 0.05
        
        t_in = np.linspace(0, duration, 3)
        q_in = np.array([
            [0, 0],
            [gy1, gz1],
            [gy2, gz2]
        ])
        qs, dqs, ts = get_cubic_control(t_in, q_in, env.dt)
        
        # Execute action
        pad_steps = int(0.2 / env.dt)
        n_steps = min(env.max_steps, len(qs) + pad_steps + 20)
        
        print(f"   Duration: {duration:.2f}s")
        print(f"   Trajectory: [0,0] → [{gy1:.3f}, {gz1:.3f}] → [{gy2:.3f}, {gz2:.3f}]")
        print(f"   Steps: {n_steps}")
        
        hist = []
        
        for i in range(n_steps):
            # Control
            ii = max(min(i, len(qs) - 1), 0)
            q = qs[ii]
            dq = dqs[ii]
            u = env.ctrl.generate(q, dq)
            env.ctrl.send_forces(u)
            sim.step()
            
            # Record keypoints (9 cloth keypoints)
            kp_com = get_body_center_of_mass(sim.data, env.kp_ids)
            hist.append(kp_com)
        
        hist = np.array(hist)  # Shape: (n_steps, 9, 3)
        
        # Compute loss
        final_cloth_pos = hist[-1]  # Shape: (9, 3)
        diff = final_cloth_pos - goal_coords
        dists = np.linalg.norm(diff, axis=-1)
        loss = np.mean(dists)
        
        print(f"   ✅ Simulated loss: {loss:.6f} ({loss * 100:.1f}cm)")
        
        # Store data
        all_trajectories.append(hist)
        all_losses.append(loss)
        all_actions_info.append({
            'action_idx': action_idx,
            'raw_action': raw_action.tolist(),
            'duration': duration,
            'n_steps': n_steps,
            'loss': loss
        })
    
    # Compute final episode loss
    final_loss = all_losses[-1]
    
    print(f"\n📊 Episode summary:")
    print(f"   Logged loss: {logged_loss:.6f} ({logged_loss * 100:.2f}cm)")
    print(f"   Replayed final loss: {final_loss:.6f} ({final_loss * 100:.2f}cm)")
    print(f"   Difference: {abs(final_loss - logged_loss):.6f} ({abs(final_loss - logged_loss) * 100:.2f}cm)")
    
    if abs(final_loss - logged_loss) < 0.001:  # Within 1mm
        print(f"   ✅ EXCELLENT MATCH!")
    elif abs(final_loss - logged_loss) < 0.01:  # Within 1cm
        print(f"   ✅ Good match")
    else:
        print(f"   ⚠️  Discrepancy detected")
    
    # Save data
    if output_data is None:
        output_data = pathlib.Path(action_log_path).with_suffix('.pkl')
    output_data = str(output_data)
    
    save_data = {
        'episode_id': episode_id,
        'trajectories': all_trajectories,  # List of arrays, each (n_steps, 9, 3)
        'losses': all_losses,
        'actions_info': all_actions_info,
        'goal_coords': goal_coords,
        'rope_config': rope_config,
        'logged_loss': logged_loss,
        'replayed_final_loss': final_loss
    }
    
    print(f"\n💾 Saving trajectory data...")
    with open(output_data, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Data saved: {output_data}")
    print(f"\nNext steps:")
    print(f"  1. Use this data to create visualization")
    print(f"  2. Generate video or screenshots from trajectories")
    print(f"  3. Create plots showing cloth manipulation")
    
    return output_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_cloth_headless.py <action_log.json> [output.pkl]")
        sys.exit(1)
    
    action_log_path = sys.argv[1]
    output_data = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        replay_episode_headless(action_log_path, output_data)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
