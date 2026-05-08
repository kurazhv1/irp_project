#!/usr/bin/env python3
"""
Modified TableClothSimEnvironment that records ALL intermediate states.
This version monkey-patches the sim.step() to capture every physics step.
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


# Global state recording
_all_states = []
_recording = False


def create_patched_environment(metadata):
    """Create environment with patched sim.step() to record states"""
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
    
    # Store reference to original sim for patching
    env._original_sim = env.sim
    
    return env, rope_config


def load_action_log(log_path):
    """Load action log from JSON"""
    with open(log_path, 'r') as f:
        return json.load(f)


def record_all_states(env, actions, save_every_n=3):
    """
    Record ALL physics states during simulation.
    
    Args:
        env: Patched TableClothSimEnvironment
        actions: List of actions
        save_every_n: Save every Nth physics step (to reduce memory)
    
    Returns:
        states: List of all qpos states
        action_boundaries: Frame indices where actions end
    """
    global _all_states, _recording
    _all_states = []
    _recording = True
    
    print(f"\n🎬 Recording ALL physics states (every {save_every_n} steps)...")
    
    # Save initial state
    _all_states.append(env.sim.data.qpos.copy())
    
    action_boundaries = []
    losses = []
    
    # Counter for downsampling
    step_counter = [0]
    
    for i, action in enumerate(tqdm(actions, desc="Recording")):
        try:
            action = np.array(action)
            
            # HACK: Manually execute env.step() logic to capture intermediate states
            # We need to replicate what env.step() does but record states
            from environments.table_cloth_sim_environment import get_cubic_control
            from abr_control_mod.mujoco_utils import get_body_center_of_mass
            
            # Generate control trajectory
            dt = env.dt
            raw_action = env.action_mapper(action)
            duration, gy1, gz1, gy2 = raw_action
            gz2 = 0.05
            t_in = np.linspace(0, duration, 3)
            q_in = np.array([[0, 0], [gy1, gz1], [gy2, gz2]])
            qs, dqs, ts = get_cubic_control(t_in, q_in, dt)
            pad_steps = int(0.2 / dt)
            
            # Reset controller
            env.ctrl._load_state(*env.init_state)
            n_steps = min(env.max_steps, len(qs) + pad_steps + 20)
            
            # Simulate step-by-step
            for step_i in range(n_steps):
                ii = max(min(step_i, len(qs)-1), 0)
                q = qs[ii]
                dq = dqs[ii]
                u = env.ctrl.generate(q, dq)
                env.ctrl.send_forces(u)
                env.sim.step()
                
                # Record state (downsampled)
                step_counter[0] += 1
                if step_counter[0] % save_every_n == 0:
                    _all_states.append(env.sim.data.qpos.copy())
            
            # Calculate loss
            kp_com = get_body_center_of_mass(env.sim.data, env.kp_ids)
            hist = np.array([kp_com])
            loss = env.loss_func(hist) if env.loss_func else 0
            
            losses.append(loss)
            action_boundaries.append(len(_all_states) - 1)
            
        except mj.MujocoException as e:
            tqdm.write(f"\n⚠️  MuJoCo error at action {i+1}: {e}")
            tqdm.write(f"   Stopping here")
            break
        except Exception as e:
            tqdm.write(f"\n❌ Error at action {i+1}: {e}")
            break
    
    _recording = False
    
    print(f"\n✅ Recorded {len(_all_states)} physics states!")
    print(f"   Actions completed: {len(action_boundaries)}/{len(actions)}")
    if action_boundaries:
        print(f"   Average states per action: {len(_all_states)/len(action_boundaries):.1f}")
    
    return _all_states, action_boundaries, losses


def main():
    parser = argparse.ArgumentParser(
        description='Record ALL cloth physics states (complete simulation)'
    )
    parser.add_argument('action_log', type=str, help='Path to action log JSON')
    parser.add_argument('--output', '-o', type=str, default='cloth_states_full.npz',
                       help='Output file (.npz)')
    parser.add_argument('--downsample', type=int, default=3,
                       help='Save every Nth physics step (default: 3)')
    args = parser.parse_args()
    
    # Load action log
    log_path = pathlib.Path(args.action_log)
    if not log_path.exists():
        print(f"❌ File not found: {log_path}")
        return 1
    
    print("=" * 70)
    print("  🎬 COMPLETE CLOTH PHYSICS RECORDER")
    print("  Records EVERY physics step (no interpolation needed!)")
    print("=" * 70)
    print(f"\n📂 Loading: {log_path}")
    
    action_log = load_action_log(log_path)
    metadata = action_log['metadata']
    actions = action_log['raw_actions']
    
    print(f"\n📋 Episode: {action_log.get('episode_id', 'unknown')}")
    print(f"   Rope ID: {metadata['rope_id']}")
    print(f"   Actions: {len(actions)}")
    print(f"   Downsampling: 1/{args.downsample} (save every {args.downsample} physics steps)")
    
    # Create patched environment
    print(f"\n🔧 Creating environment with state recording...")
    env, rope_config = create_patched_environment(metadata)
    print(f"   ✓ Environment ready (sim.step() patched)")
    
    # Record all states
    states, action_boundaries, losses = record_all_states(env, actions, args.downsample)
    
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
            'downsampling': args.downsample,
        }
    )
    
    print(f"   ✅ Saved {len(states)} states!")
    print(f"\n💡 To view: python replay_states_viewer.py {output_path}")
    print("   (No interpolation needed - real physics!)")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
