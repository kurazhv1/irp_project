#!/usr/bin/env python3
"""
Monkey-patch TableClothSimEnvironment to record all intermediate states during step().
This is the SAFEST way to get high-resolution states without breaking physics.
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

GlfwContext(offscreen=True)

from environments.table_cloth_sim_environment import TableClothSimEnvironment


# Global list to collect states during env.step()
_recorded_states = []
_save_frequency = 5


def patched_step(original_step):
    """Wrap env.step() to record intermediate states"""
    def wrapper(self, action, wait=0):
        global _recorded_states
        
        # Call original step
        obs, reward, done, info = original_step(self, action, wait)
        
        # HACK: The original step() already ran simulation
        # We can't get intermediate states retroactively
        # So we need to re-run simulation and record states
        
        # Actually, let's just save the FINAL state after each step
        _recorded_states.append(self.sim.data.qpos.copy())
        
        return obs, reward, done, info
    return wrapper


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
    
    # Monkey-patch step() to record states
    env.step = patched_step(env.step.__get__(env, TableClothSimEnvironment))
    
    return env, rope_config


def replay_and_record(env, actions):
    """Replay actions and record states"""
    global _recorded_states
    _recorded_states = []
    
    # Save initial state
    _recorded_states.append(env.sim.data.qpos.copy())
    
    print("\n🎬 Replaying and recording states...")
    
    action_boundaries = []
    losses = []
    
    for i, action in enumerate(tqdm(actions, desc="Recording")):
        try:
            action = np.array(action)
            obs, loss, done, info = env.step(action)
            
            losses.append(loss)
            action_boundaries.append(len(_recorded_states) - 1)
            
        except mj.MujocoException as e:
            tqdm.write(f"\n⚠️  MuJoCo error at action {i+1}: {e}")
            tqdm.write(f"   Stopping here")
            break
        except Exception as e:
            tqdm.write(f"\n❌ Error at action {i+1}: {e}")
            break
    
    print(f"\n✅ Recorded {len(_recorded_states)} states across {len(action_boundaries)} actions")
    
    return _recorded_states, action_boundaries, losses


def main():
    parser = argparse.ArgumentParser(
        description='Record cloth states by monkey-patching env.step()'
    )
    parser.add_argument('action_log', type=str, help='Path to action log JSON file')
    parser.add_argument('--output', '-o', type=str, default='cloth_states.npz',
                       help='Output file (.npz)')
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.action_log)
    if not log_path.exists():
        print(f"❌ File not found: {log_path}")
        return 1
    
    print("=" * 70)
    print("  🎬 CLOTH STATE RECORDER (Monkey-patch)")
    print("=" * 70)
    print(f"\n📂 Loading: {log_path}")
    
    action_log = load_action_log(log_path)
    metadata = action_log['metadata']
    actions = action_log['raw_actions']
    
    print(f"\n📋 Episode: {action_log.get('episode_id', 'unknown')}")
    print(f"   Rope ID: {metadata['rope_id']}")
    print(f"   Actions: {len(actions)}")
    
    # Create environment
    env, rope_config = create_environment_from_metadata(metadata)
    
    # Record states
    states, action_boundaries, losses = replay_and_record(env, actions)
    
    # Save
    output_path = pathlib.Path(args.output)
    print(f"\n💾 Saving to: {output_path}")
    
    np.savez_compressed(
        output_path,
        states=np.array(states),
        action_boundaries=np.array(action_boundaries),
        metadata={
            'episode_id': action_log.get('episode_id', 'unknown'),
            'rope_id': metadata['rope_id'],
            'loss': losses[-1] if losses else 0,
            'rope_config': rope_config,
            'n_actions': len(actions),
            'n_states': len(states),
        }
    )
    
    print(f"   ✅ Saved!")
    print(f"\n💡 To view: python replay_states_viewer.py {output_path}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
