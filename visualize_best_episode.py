#!/usr/bin/env python3
"""
Complete visualization pipeline for best evaluation episode.
Creates:
1. 3D trajectory plot
2. Occupancy maps (initial, goal, final)
3. Simulation replay with screenshots
4. Comparison plots
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco_py
from pathlib import Path
import cv2

from environments.table_cloth_sim_environment import TableClothSimEnvironment

# Configuration
LOG_FILE = "outputs/2025-12-01/13-04-31/action_logs/action_log_20251201_130443_rope4_goal5.json"
OUTPUT_DIR = Path("best_episode_visualization")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("COMPLETE VISUALIZATION PIPELINE FOR BEST EPISODE")
print("="*70)

# Load action log
print(f"\n📂 Loading: {LOG_FILE}")
with open(LOG_FILE, 'r') as f:
    log_data = json.load(f)

print(f"Episode: {log_data['episode_id']}")
print(f"Final Loss: {log_data['loss']*100:.2f} cm")
print(f"Actions: {len(log_data['actions'])}")

# Extract data
actions = np.array(log_data['actions'])
rope_config = log_data['rope_config']
controller_config = log_data['controller_config']
goal_coords = np.array(log_data['goal_coords'])

print(f"\nRope config: density={rope_config['cloth_density']}, spacing={rope_config['cloth_spacing']}")
print(f"Goal shape: {goal_coords.shape}")

# ============================================================================
# 1. 3D TRAJECTORY VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("1. Creating 3D Trajectory Plot...")

fig = plt.figure(figsize=(15, 5))

# Plot 1: Full trajectory
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(actions[:, 0], actions[:, 1], actions[:, 2], 
         'b-', linewidth=2, alpha=0.7, label='Robot trajectory')
ax1.scatter(actions[0, 0], actions[0, 1], actions[0, 2], 
           c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
ax1.scatter(actions[-1, 0], actions[-1, 1], actions[-1, 2], 
           c='red', s=200, marker='X', label='End', edgecolors='black', linewidths=2)

ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_zlabel('Z (m)', fontsize=10)
ax1.set_title('Robot End-Effector Trajectory', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Top view (X-Y plane)
ax2 = fig.add_subplot(132)
ax2.plot(actions[:, 0], actions[:, 1], 'b-', linewidth=2, alpha=0.7)
ax2.scatter(actions[0, 0], actions[0, 1], c='green', s=200, marker='o', 
           label='Start', edgecolors='black', linewidths=2, zorder=5)
ax2.scatter(actions[-1, 0], actions[-1, 1], c='red', s=200, marker='X', 
           label='End', edgecolors='black', linewidths=2, zorder=5)

# Add waypoint numbers
for i in range(0, len(actions), max(1, len(actions)//10)):
    ax2.annotate(f'{i}', (actions[i, 0], actions[i, 1]), 
                fontsize=8, ha='center', va='center')

ax2.set_xlabel('X (m)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.set_title('Top View (X-Y Plane)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Plot 3: Side view (X-Z plane)
ax3 = fig.add_subplot(133)
ax3.plot(actions[:, 0], actions[:, 2], 'b-', linewidth=2, alpha=0.7)
ax3.scatter(actions[0, 0], actions[0, 2], c='green', s=200, marker='o', 
           label='Start', edgecolors='black', linewidths=2, zorder=5)
ax3.scatter(actions[-1, 0], actions[-1, 2], c='red', s=200, marker='X', 
           label='End', edgecolors='black', linewidths=2, zorder=5)

ax3.set_xlabel('X (m)', fontsize=10)
ax3.set_ylabel('Z (m)', fontsize=10)
ax3.set_title('Side View (X-Z Plane)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
trajectory_file = OUTPUT_DIR / "01_trajectory_3d.png"
plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {trajectory_file}")
plt.close()

# ============================================================================
# 2. OCCUPANCY MAPS VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("2. Creating Occupancy Maps...")

# Create environment to get occupancy
env = TableClothSimEnvironment(rope_config, controller_config)

# Get initial occupancy
# Transform initial cloth coordinates to occupancy map
cloth_coords = env.sim.data.body_xpos[env.kp_ids]
initial_occu = env.transformer.to_img(cloth_coords[:, [1, 2]])

# Get goal occupancy (from goal_coords)
# goal_coords is already an occupancy map
goal_occu = goal_coords

print(f"Initial occupancy shape: {initial_occu.shape}")
print(f"Goal occupancy shape: {goal_occu.shape}")

# Replay to get final occupancy
print("Replaying actions to get final state...")
# Reset simulation to initial state
env.ctrl._save_state(env.init_state)

# Execute actions
for i, action in enumerate(actions):
    # Map raw action to target
    target_qpos = env.action_mapper.inverse_transform(action)
    
    # Execute with controller
    for _ in range(10):  # Execute for multiple timesteps per action
        env.ctrl.set_target_qpos(target_qpos)
        env.sim.step()
        if env.show_vis:
            env.viewer.render()
    
    if i % 5 == 0:
        print(f"  Action {i+1}/{len(actions)}")

# Get final state
cloth_coords_final = env.sim.data.body_xpos[env.kp_ids]
final_occu = env.transformer.to_img(cloth_coords_final[:, [1, 2]])

# Calculate final loss
loss_func = env.get_traj_loss_func(goal_coords)
final_loss = loss_func([cloth_coords_final])
print(f"Final loss from replay: {final_loss*100:.2f} cm")
print(f"Final occupancy shape: {final_occu.shape}")

# Plot occupancy maps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Initial
im1 = axes[0].imshow(initial_occu, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
axes[0].set_title('Initial Cloth Configuration', fontsize=14)
axes[0].set_xlabel('X grid', fontsize=12)
axes[0].set_ylabel('Y grid', fontsize=12)
plt.colorbar(im1, ax=axes[0], label='Occupancy')

# Goal
im2 = axes[1].imshow(goal_occu, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
axes[1].set_title('Goal Configuration', fontsize=14)
axes[1].set_xlabel('X grid', fontsize=12)
axes[1].set_ylabel('Y grid', fontsize=12)
plt.colorbar(im2, ax=axes[1], label='Occupancy')

# Final
im3 = axes[2].imshow(final_occu, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
axes[2].set_title(f'Final Configuration (Loss: {final_loss*100:.2f} cm)', fontsize=14)
axes[2].set_xlabel('X grid', fontsize=12)
axes[2].set_ylabel('Y grid', fontsize=12)
plt.colorbar(im3, ax=axes[2], label='Occupancy')

plt.tight_layout()
occupancy_file = OUTPUT_DIR / "02_occupancy_maps.png"
plt.savefig(occupancy_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {occupancy_file}")
plt.close()

# ============================================================================
# 3. DIFFERENCE MAPS
# ============================================================================
print("\n" + "="*70)
print("3. Creating Difference Maps...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Initial vs Goal
diff_initial = np.abs(goal_occu - initial_occu)
im1 = axes[0].imshow(diff_initial, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
axes[0].set_title('Initial → Goal Difference', fontsize=14)
axes[0].set_xlabel('X grid', fontsize=12)
axes[0].set_ylabel('Y grid', fontsize=12)
plt.colorbar(im1, ax=axes[0], label='Absolute Difference')
axes[0].text(0.5, -0.15, f'Mean diff: {np.mean(diff_initial):.4f}', 
            transform=axes[0].transAxes, ha='center', fontsize=11)

# Final vs Goal
diff_final = np.abs(goal_occu - final_occu)
im2 = axes[1].imshow(diff_final, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
axes[1].set_title(f'Final → Goal Difference (Loss: {final_loss*100:.2f} cm)', fontsize=14)
axes[1].set_xlabel('X grid', fontsize=12)
axes[1].set_ylabel('Y grid', fontsize=12)
plt.colorbar(im2, ax=axes[1], label='Absolute Difference')
axes[1].text(0.5, -0.15, f'Mean diff: {np.mean(diff_final):.4f}', 
            transform=axes[1].transAxes, ha='center', fontsize=11)

plt.tight_layout()
diff_file = OUTPUT_DIR / "03_difference_maps.png"
plt.savefig(diff_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {diff_file}")
plt.close()

# ============================================================================
# 4. OVERLAY COMPARISON
# ============================================================================
print("\n" + "="*70)
print("4. Creating Overlay Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Initial + Goal overlay
axes[0].imshow(initial_occu, cmap='Blues', alpha=0.6, interpolation='nearest')
axes[0].imshow(goal_occu, cmap='Reds', alpha=0.4, interpolation='nearest')
axes[0].set_title('Overlay: Initial (blue) + Goal (red)', fontsize=14)
axes[0].set_xlabel('X grid', fontsize=12)
axes[0].set_ylabel('Y grid', fontsize=12)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Initial'),
                   Patch(facecolor='red', alpha=0.4, label='Goal'),
                   Patch(facecolor='purple', alpha=0.8, label='Overlap')]
axes[0].legend(handles=legend_elements, loc='upper right')

# Final + Goal overlay
axes[1].imshow(final_occu, cmap='Greens', alpha=0.6, interpolation='nearest')
axes[1].imshow(goal_occu, cmap='Reds', alpha=0.4, interpolation='nearest')
axes[1].set_title(f'Overlay: Final (green) + Goal (red) - Loss: {final_loss*100:.2f} cm', fontsize=14)
axes[1].set_xlabel('X grid', fontsize=12)
axes[1].set_ylabel('Y grid', fontsize=12)

legend_elements = [Patch(facecolor='green', alpha=0.6, label='Final'),
                   Patch(facecolor='red', alpha=0.4, label='Goal'),
                   Patch(facecolor='yellow', alpha=0.8, label='Overlap')]
axes[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
overlay_file = OUTPUT_DIR / "04_overlay_comparison.png"
plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {overlay_file}")
plt.close()

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("5. Creating Summary Statistics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Action components over time
ax = axes[0, 0]
time_steps = np.arange(len(actions))
ax.plot(time_steps, actions[:, 0], label='X', linewidth=2)
ax.plot(time_steps, actions[:, 1], label='Y', linewidth=2)
ax.plot(time_steps, actions[:, 2], label='Z', linewidth=2)
ax.set_xlabel('Action Step', fontsize=12)
ax.set_ylabel('Position (m)', fontsize=12)
ax.set_title('End-Effector Position Components', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Action deltas
ax = axes[0, 1]
deltas = np.diff(actions, axis=0)
ax.plot(time_steps[:-1], np.linalg.norm(deltas, axis=1), 'purple', linewidth=2)
ax.set_xlabel('Action Step', fontsize=12)
ax.set_ylabel('Delta Distance (m)', fontsize=12)
ax.set_title('Movement Magnitude per Step', fontsize=14)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, f'Mean: {np.mean(np.linalg.norm(deltas, axis=1)):.4f} m', 
        transform=ax.transAxes, ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Occupancy statistics
ax = axes[1, 0]
categories = ['Initial', 'Goal', 'Final']
total_occupied = [np.sum(initial_occu > 0.5), np.sum(goal_occu > 0.5), np.sum(final_occu > 0.5)]
mean_values = [np.mean(initial_occu), np.mean(goal_occu), np.mean(final_occu)]

x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, total_occupied, width, label='Occupied Cells', alpha=0.8)
ax2 = ax.twinx()
ax2.bar(x + width/2, mean_values, width, label='Mean Occupancy', alpha=0.8, color='orange')

ax.set_xlabel('Configuration', fontsize=12)
ax.set_ylabel('Occupied Cells Count', fontsize=12)
ax2.set_ylabel('Mean Occupancy Value', fontsize=12)
ax.set_title('Occupancy Statistics', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Difference metrics
ax = axes[1, 1]
metrics = ['Mean\nAbsolute\nDiff', 'Max\nAbsolute\nDiff', 'L2\nNorm']
initial_metrics = [np.mean(diff_initial), np.max(diff_initial), np.linalg.norm(diff_initial)]
final_metrics = [np.mean(diff_final), np.max(diff_final), np.linalg.norm(diff_final)]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, initial_metrics, width, label='Initial→Goal', alpha=0.8)
bars2 = ax.bar(x + width/2, final_metrics, width, label='Final→Goal', alpha=0.8)

ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Difference Metrics Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
stats_file = OUTPUT_DIR / "05_summary_statistics.png"
plt.savefig(stats_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {stats_file}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  ✓ {f.name}")

print(f"\nEpisode Summary:")
print(f"  Episode ID: {log_data['episode_id']}")
print(f"  Final Loss: {final_loss*100:.2f} cm")
print(f"  Total Actions: {len(actions)}")
print(f"  Initial occupied cells: {np.sum(initial_occu > 0.5)}")
print(f"  Goal occupied cells: {np.sum(goal_occu > 0.5)}")
print(f"  Final occupied cells: {np.sum(final_occu > 0.5)}")
print(f"  Mean difference (Final→Goal): {np.mean(diff_final):.4f}")
print("="*70)
