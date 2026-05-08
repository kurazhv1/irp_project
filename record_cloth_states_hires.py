#!/usr/bin/env python3
"""
Record cloth simulation states with HIGH temporal resolution.
This version manually steps through simulation to capture ALL intermediate states.
"""

import json
import pathlib
import argparse
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import mujoco_py as mj
from mujoco_py import GlfwContext
from scipy.interpolate import CubicSpline

# Initialize GLFW for offscreen rendering
GlfwContext(offscreen=True)

from environments.table_cloth_sim_environment import TableClothSimEnvironment, ActionMapper, get_cubic_control


def load_action_log(log_path):
    """Load action log from JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def create_environment_from_metadata(metadata):
    """Create TableClothSimEnvironment from metadata"""
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
    
    env = TableClothSimEnvironment(
        rope_config=rope_config,
        controller_config=controller_config,
        obs_topdown=False,
        show_vis=False
    )
    
    goal_alpha = metadata.get('goal_alpha', 0.0)
    goal = env.get_cloth_goal(goal_alpha)
    loss_func = env.get_traj_loss_func(goal, measure_dims=[0, 1, 2])
    env.set_loss_func(loss_func)
    
    return env, rope_config


def record_detailed_states(env, actions, save_every_n_steps=5):
    """
    Record cloth states with high temporal resolution.
    
    Args:
        env: TableClothSimEnvironment
        actions: List of actions [duration, gy1, gz1, gy2]
        save_every_n_steps: Save state every N simulation steps (default: 5)
    
    Returns:
        all_states: List of qpos states
        action_boundaries: List of indices where each action ends
    """
    print("\n🎬 Recording high-resolution cloth states...")
    print(f"   Save frequency: every {save_every_n_steps} simulation steps")
    
    all_states = []
    action_boundaries = []
    
    # Save initial state
    all_states.append(env.sim.data.qpos.copy())
    
    action_mapper = ActionMapper()
    dt = env.dt
    
    for i, action in enumerate(tqdm(actions, desc="Recording")):
        try:
            # Convert action
            action = np.array(action)
            raw_action = action_mapper(action)
            duration, gy1, gz1, gy2 = raw_action
            gz2 = 0.05
            
            # Generate cubic spline trajectory
            t_in = np.linspace(0, duration, 3)
            q_in = np.array([
                [0, 0],
                [gy1, gz1],
                [gy2, gz2]
            ])
            qs, dqs, ts = get_cubic_control(t_in, q_in, dt)
            
            pad_steps = int(0.2 / dt)
            n_steps = min(env.max_steps, len(qs) + pad_steps + 20)
            
            # Reset controller to initial state
            env.ctrl._load_state(*env.init_state)
            
            # Simulate action step-by-step
            step_counter = 0
            for step_i in range(n_steps):
                ii = max(min(step_i, len(qs)-1), 0)
                q = qs[ii]
                dq = dqs[ii]
                
                # Generate and apply control
                u = env.ctrl.generate(q, dq)
                env.ctrl.send_forces(u)
                env.sim.step()
                
                # Save state at specified frequency
                if step_counter % save_every_n_steps == 0:
                    all_states.append(env.sim.data.qpos.copy())
                
                step_counter += 1
            
            # Mark action boundary
            action_boundaries.append(len(all_states) - 1)
            
        except mj.MujocoException as e:
            tqdm.write(f"\n⚠️  MuJoCo error at action {i+1}: {e}")
            tqdm.write(f"   Stopping here (this is normal for unstable cloth)")
            break
        except Exception as e:
            tqdm.write(f"\n❌ Error at action {i+1}: {e}")
            break
    
    print(f"\n✅ Recorded {len(all_states)} states across {len(action_boundaries)} actions")
    print(f"   Average: {len(all_states)/max(len(action_boundaries), 1):.1f} states per action")
    
    return all_states, action_boundaries


def main():
    parser = argparse.ArgumentParser(
        description='Record high-resolution cloth simulation states'
    )
    parser.add_argument('action_log', type=str, help='Path to action log JSON file')
    parser.add_argument('--output', '-o', type=str, default='cloth_states_hires.npz',
                       help='Output file for states (.npz)')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save state every N simulation steps (default: 5)')
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.action_log)
    if not log_path.exists():
        print(f"❌ ERROR: File not found: {log_path}")
        return 1
    
    print("=" * 70)
    print("  🎬 HIGH-RESOLUTION CLOTH STATE RECORDER")
    print("=" * 70)
    print(f"\n📂 Loading: {log_path}")
    
    action_log = load_action_log(log_path)
    metadata = action_log['metadata']
    actions = action_log['raw_actions']
    
    print(f"\n📋 Episode: {action_log.get('episode_id', 'unknown')}")
    print(f"   Rope ID: {metadata['rope_id']}")
    print(f"   Actions: {len(actions)}")
    print(f"   Final loss: {action_log.get('loss', 0):.4f}")
    
    # Create environment
    print(f"\n🔧 Creating environment...")
    env, rope_config = create_environment_from_metadata(metadata)
    print(f"   ✓ Environment ready")
    
    # Record states
    states, action_boundaries = record_detailed_states(env, actions, args.save_every)
    
    # Save to file
    output_path = pathlib.Path(args.output)
    print(f"\n💾 Saving to: {output_path}")
    
    np.savez_compressed(
        output_path,
        states=np.array(states),
        action_boundaries=np.array(action_boundaries),
        metadata={
            'episode_id': action_log.get('episode_id', 'unknown'),
            'rope_id': metadata['rope_id'],
            'loss': action_log.get('loss', 0),
            'rope_config': rope_config,
            'n_actions': len(actions),
            'n_states': len(states),
            'save_frequency': args.save_every,
        }
    )
    
    print(f"   ✅ Saved successfully!")
    print(f"\n💡 To view: python replay_states_viewer.py {output_path}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
